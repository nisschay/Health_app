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
  signInWithPopup,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  updateProfile,
  GoogleAuthProvider,
  signOut,
} from "firebase/auth";
import { auth } from "./firebase";
import { buildApiUrl, getDirectApiBaseUrl, getPublicApiBaseUrl } from "./apiBaseUrl";

const HF_SPACE_BACKEND_URL = "https://nisschay-medical-project-backend.hf.space";

const AUTH_PRESENCE_COOKIE = "mra_auth";
const AUTH_PRESENCE_MAX_AGE_SECONDS = 60 * 60 * 12;

function setAuthPresenceCookie(): void {
  if (typeof document === "undefined") return;
  document.cookie = `${AUTH_PRESENCE_COOKIE}=1; Path=/; Max-Age=${AUTH_PRESENCE_MAX_AGE_SECONDS}; SameSite=Lax`;
}

function clearAuthPresenceCookie(): void {
  if (typeof document === "undefined") return;
  document.cookie = `${AUTH_PRESENCE_COOKIE}=; Path=/; Max-Age=0; SameSite=Lax`;
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

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [isAdmin, setIsAdmin] = useState(false);

  useEffect(() => {
    let isActive = true;

    const unsub = onAuthStateChanged(auth, (firebaseUser) => {
      setUser(firebaseUser);
      if (!firebaseUser) {
        clearAuthPresenceCookie();
        setIsAdmin(false);
        setLoading(false);
        return;
      }

      setAuthPresenceCookie();

      setLoading(true);

      // Sync user to backend PostgreSQL on sign-in
      firebaseUser.getIdToken()
        .then((token) => {
          const publicApiBase = getPublicApiBaseUrl();
          const syncBase = process.env.NODE_ENV === "production"
            ? HF_SPACE_BACKEND_URL
            : (publicApiBase.startsWith("/") ? getDirectApiBaseUrl() : publicApiBase);
          const syncUrl = buildApiUrl(syncBase, "/api/v1/auth/sync", {
            display_name: firebaseUser.displayName ?? "",
          });
          return fetch(
            syncUrl,
            {
              method: "POST",
              headers: { Authorization: `Bearer ${token}` },
            },
          );
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
    await signInWithPopup(auth, googleProvider);
    setAuthPresenceCookie();
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
