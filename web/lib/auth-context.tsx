"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
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

const AUTH_PRESENCE_COOKIE = "mra_auth";
const AUTH_PRESENCE_MAX_AGE_SECONDS = 60 * 60 * 12;
const BACKEND_SYNC_TIMEOUT_MS = 5_000;

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

/** Upserts the signed-in user and their own profile in the backend. A failure is retried on the next sign-in. */
async function syncBackendUser(firebaseUser: User): Promise<void> {
  const syncAbort = new AbortController();
  const syncTimeout = window.setTimeout(() => syncAbort.abort(), BACKEND_SYNC_TIMEOUT_MS);
  try {
    const token = await firebaseUser.getIdToken();
    const publicApiBase = getPublicApiBaseUrl();
    const directApiBase = getDirectApiBaseUrl();
    const primaryBase = publicApiBase.startsWith("/") ? directApiBase : publicApiBase;
    const syncTargets = [primaryBase, directApiBase].filter(
      (value, index, arr) => Boolean(value) && arr.indexOf(value) === index,
    );
    for (const base of syncTargets) {
      const response = await fetch(
        buildApiUrl(base!, "/api/v1/auth/sync", { display_name: firebaseUser.displayName ?? "" }),
        { method: "POST", headers: { Authorization: `Bearer ${token}` }, signal: syncAbort.signal },
      );
      if (response.status !== 404 && response.status < 500) return;
    }
  } catch {
    // the backend also upserts on every route
  } finally {
    window.clearTimeout(syncTimeout);
  }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

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

      setLoading(false);

      void syncBackendUser(firebaseUser);
    });

    return () => {
      isActive = false;
      unsub();
    };
  }, []);

  const getToken = useCallback(async (forceRefresh = false): Promise<string | null> => {
    if (!user) return null;
    try {
      return await user.getIdToken(forceRefresh);
    } catch (error) {
      if (!forceRefresh) {
        return user.getIdToken(true);
      }
      throw error;
    }
  }, [user]);

  const signInWithGoogle = useCallback(async () => {
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
  }, []);

  const signInWithEmail = useCallback(async (email: string, password: string) => {
    await signInWithEmailAndPassword(auth, email, password);
    setAuthPresenceCookie();
  }, []);

  const registerWithEmail = useCallback(async (email: string, password: string, displayName: string) => {
    const credential = await createUserWithEmailAndPassword(auth, email, password);
    await updateProfile(credential.user, { displayName });
    // The sign-in listener already synced before the name existed; send it now.
    await syncBackendUser(credential.user);
    setAuthPresenceCookie();
  }, []);

  const logout = useCallback(async () => {
    try {
      await signOut(auth);
    } finally {
      clearAuthPresenceCookie();
      if (typeof window !== "undefined") {
        window.localStorage.clear();
        window.sessionStorage.clear();
      }
    }
  }, []);

  // A fresh object every render re-fired every effect that listed getToken or logout.
  const value = useMemo<AuthContextValue>(
    () => ({ user, loading, getToken, signInWithGoogle, signInWithEmail, registerWithEmail, logout }),
    [user, loading, getToken, signInWithGoogle, signInWithEmail, registerWithEmail, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}
