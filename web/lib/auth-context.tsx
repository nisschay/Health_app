"use client";

import {
  createContext,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import {
  type User,
  onAuthStateChanged,
  getRedirectResult,
  signInWithPopup,
  signInWithRedirect,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  updateProfile,
  GoogleAuthProvider,
  signOut,
} from "firebase/auth";
import { auth } from "./firebase";
import { buildApiUrl, getDirectApiBaseUrl, getPublicApiBaseUrl } from "./apiBaseUrl";
import { setAuthTokenProvider } from "./api";

// Presence hint only: it tells middleware whether to bother rendering a
// protected route. It proves nothing. Every request is authorised by the
// Firebase ID token the backend verifies.
const AUTH_PRESENCE_COOKIE = "mra_auth";
const AUTH_PRESENCE_MAX_AGE_SECONDS = 60 * 60 * 12;

function cookieFlags(): string {
  const secure = typeof location !== "undefined" && location.protocol === "https:" ? "; Secure" : "";
  return `Path=/; SameSite=Lax${secure}`;
}

function setAuthPresenceCookie(): void {
  if (typeof document === "undefined") return;
  document.cookie = `${AUTH_PRESENCE_COOKIE}=1; ${cookieFlags()}; Max-Age=${AUTH_PRESENCE_MAX_AGE_SECONDS}`;
}

function clearAuthPresenceCookie(): void {
  if (typeof document === "undefined") return;
  document.cookie = `${AUTH_PRESENCE_COOKIE}=; ${cookieFlags()}; Max-Age=0`;
}

export type AuthContextValue = {
  user: User | null;
  loading: boolean;
  isAdmin: boolean;
  getToken: (forceRefresh?: boolean) => Promise<string | null>;
  signInWithGoogle: () => Promise<void>;
  signInWithEmail: (email: string, password: string) => Promise<void>;
  registerWithEmail: (email: string, password: string, displayName: string) => Promise<void>;
  logout: () => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | null>(null);

const googleProvider = new GoogleAuthProvider();
googleProvider.setCustomParameters({ prompt: "select_account" });

function shouldFallBackToRedirect(error: unknown): boolean {
  if (!(error instanceof Error)) {
    return false;
  }

  return (
    error.message.includes("auth/popup-blocked")
    || error.message.includes("auth/popup-closed-by-user")
    || error.message.includes("auth/cancelled-popup-request")
    || error.message.includes("auth/operation-not-supported-in-this-environment")
  );
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [isAdmin, setIsAdmin] = useState(false);

  useEffect(() => {
    let isActive = true;

    getRedirectResult(auth)
      .then((result) => {
        if (!isActive || !result?.user) {
          return;
        }
        setAuthPresenceCookie();
      })
      .catch((error: unknown) => {
        if (!isActive) {
          return;
        }
        console.warn("[Auth] redirect sign-in failed", error);
      });

    const unsub = onAuthStateChanged(auth, (firebaseUser) => {
      setUser(firebaseUser);
      if (!firebaseUser) {
        clearAuthPresenceCookie();
        setAuthTokenProvider(async () => null);
        setIsAdmin(false);
        setLoading(false);
        return;
      }

      setAuthPresenceCookie();
      // Register token provider so api.ts can always get fresh token
      setAuthTokenProvider(async () => {
        try {
          return await firebaseUser.getIdToken(false);
        } catch {
          try {
            return await firebaseUser.getIdToken(true);
          } catch {
            return null;
          }
        }
      });

      setLoading(true);

      // Sync user to backend PostgreSQL on sign-in
      firebaseUser.getIdToken()
        .then(async (token) => {
          const publicApiBase = getPublicApiBaseUrl();
          const directApiBase = getDirectApiBaseUrl();
          const primaryBase = publicApiBase.startsWith("/") ? directApiBase : publicApiBase;

          const syncTargets = [primaryBase, directApiBase].filter(
            (value, index, arr) => Boolean(value) && arr.indexOf(value) === index,
          );

          const syncUrl = buildApiUrl(syncTargets[0]!, "/api/v1/auth/sync", {
            display_name: firebaseUser.displayName ?? "",
          });
          let response = await fetch(
            syncUrl,
            {
              method: "POST",
              headers: { Authorization: `Bearer ${token}` },
            },
          );

          if ((response.status === 404 || response.status >= 500) && syncTargets.length > 1) {
            const fallbackSyncUrl = buildApiUrl(syncTargets[1]!, "/api/v1/auth/sync", {
              display_name: firebaseUser.displayName ?? "",
            });
            response = await fetch(
              fallbackSyncUrl,
              {
                method: "POST",
                headers: { Authorization: `Bearer ${token}` },
              },
            );
          }

          return response;
        })
        .then(async (response) => {
          if (!isActive) {
            return;
          }
          if (!response.ok) {
            setIsAdmin(false);
            return;
          }
          const payload = (await response.json()) as { is_admin?: unknown };
          setIsAdmin(Boolean(payload.is_admin));
        })
        .catch((error: unknown) => {
          if (!isActive) {
            return;
          }
          console.warn("[Auth] backend sync failed", error);
          setIsAdmin(false);
        })
        .finally(() => {
          if (isActive) {
            setLoading(false);
          }
        });
    });

    return () => {
      isActive = false;
      unsub();
    };
  }, []);

  async function getToken(forceRefresh = false): Promise<string | null> {
    if (!user) return null;
    try {
      return await user.getIdToken(forceRefresh);
    } catch (error) {
      if (!forceRefresh) {
        return user.getIdToken(true);
      }
      throw error;
    }
  }

  async function signInWithGoogle() {
    try {
      const credential = await signInWithPopup(auth, googleProvider);
      if (credential.user) {
        setAuthPresenceCookie();
      }
    } catch (error) {
      if (shouldFallBackToRedirect(error)) {
        await signInWithRedirect(auth, googleProvider);
        return;
      }
      throw error;
    }
  }

  async function signInWithEmail(email: string, password: string) {
    await signInWithEmailAndPassword(auth, email, password);
    setAuthPresenceCookie();
  }

  async function registerWithEmail(
    email: string,
    password: string,
    displayName: string
  ) {
    const credential = await createUserWithEmailAndPassword(auth, email, password);
    await updateProfile(credential.user, { displayName });
    setAuthPresenceCookie();
  }

  async function logout() {
    try {
      await signOut(auth);
    } finally {
      clearAuthPresenceCookie();
      if (typeof window !== "undefined") {
        window.localStorage.clear();
        window.sessionStorage.clear();
      }
    }
  }

  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        isAdmin,
        getToken,
        signInWithGoogle,
        signInWithEmail,
        registerWithEmail,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}
