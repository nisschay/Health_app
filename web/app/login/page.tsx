"use client";
import Brand from "../Brand";

import { useState, useEffect, useRef } from "react";
import { sendPasswordResetEmail } from "firebase/auth";
import { auth } from "@/lib/firebase";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-context";

/** Where the middleware sent the user from, if it is a path on this site; else the dashboard. */
function pathAfterLogin(): string {
  const next = new URLSearchParams(window.location.search).get("next");
  return next && next.startsWith("/") && !next.startsWith("//")
    ? next
    : "/dashboard";
}

export default function LoginPage() {
  const {
    user,
    loading,
    signInWithGoogle,
    signInWithEmail,
    registerWithEmail,
  } = useAuth();
  const router = useRouter();

  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isPending, setIsPending] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const displayNameInputRef = useRef<HTMLInputElement>(null);
  const emailInputRef = useRef<HTMLInputElement>(null);
  const passwordInputRef = useRef<HTMLInputElement>(null);

  // While a form submit runs, it redirects itself when done: sign-up must save the name and sync it first,
  // or the dashboard loads the profile under the email-made placeholder.
  useEffect(() => {
    if (!loading && user && !isPending) {
      router.replace(pathAfterLogin());
    }
  }, [user, loading, router, isPending]);

  function handleModeChange(nextMode: "login" | "register") {
    setMode(nextMode);
    setDisplayName("");
    setEmail("");
    setPassword("");
    setError(null);
    setShowPassword(false);
  }

  async function handleEmailSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setIsPending(true);
    try {
      if (mode === "login") {
        await signInWithEmail(
          emailInputRef.current?.value || email,
          passwordInputRef.current?.value || password,
        );
      } else {
        if (!displayName.trim()) {
          setError("Please enter your full name.");
          setIsPending(false);
          return;
        }
        await registerWithEmail(email, password, displayName);
      }
      router.replace(pathAfterLogin());
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Authentication failed.";
      // Surface friendly messages
      if (
        msg.includes("user-not-found") ||
        msg.includes("wrong-password") ||
        msg.includes("invalid-credential")
      ) {
        setError("Invalid email or password.");
      } else if (msg.includes("email-already-in-use")) {
        setError("An account with this email already exists.");
      } else if (msg.includes("weak-password")) {
        setError("Password must be at least 6 characters.");
      } else {
        setError(msg);
      }
    } finally {
      setIsPending(false);
    }
  }

  async function handleGoogle() {
    setError(null);
    setIsPending(true);
    try {
      await signInWithGoogle();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Google sign-in failed.";
      if (!msg.includes("popup-closed")) {
        setError(msg);
      }
    } finally {
      setIsPending(false);
    }
  }

  if (loading) {
    return (
      <main className="auth">
        <div className="auth-loading">Loading…</div>
      </main>
    );
  }

  return (
    <main className="auth">
      <section className="auth-story" aria-hidden="true">
        <h2 className="display">
          Follow your family&apos;s lab results over time.
        </h2>
        <p>
          Add the PDF from the lab. See what is outside the normal range, how it
          has changed, and what to ask your doctor.
        </p>
        <div className="sample">
          <strong>HbA1c</strong>
          <strong>7.9 %</strong>
          <span>Getting better · down 0.3 since 2 Nov</span>
          <span className="chip high">↑ High</span>
        </div>
        <ul className="auth-points">
          <li>Keep reports for parents, partner and children in one place</li>
          <li>See each test over time, against the lab&apos;s own range</li>
          <li>Your reports stay yours. Download or delete them at any time</li>
        </ul>
      </section>

      <section className="auth-form-side">
        <div className="auth-card">
          <div className="auth-brand brand">
            <Brand />
          </div>
          <h1 className="h1">
            {mode === "login" ? "Welcome back" : "Create your account"}
          </h1>
          <div className="tabs" role="tablist" aria-label="Authentication mode">
            <button
              className={`auth-tab ${mode === "login" ? "active" : ""}`}
              onClick={() => handleModeChange("login")}
              type="button"
              role="tab"
              aria-selected={mode === "login"}
            >
              Sign in
            </button>
            <button
              className={`auth-tab ${mode === "register" ? "active" : ""}`}
              onClick={() => handleModeChange("register")}
              type="button"
              role="tab"
              aria-selected={mode === "register"}
            >
              Create account
            </button>
          </div>

          <button
            className="google-button"
            disabled={isPending}
            onClick={handleGoogle}
            type="button"
          >
            <span className="google-glyph" aria-hidden="true">
              G
            </span>
            Continue with Google
          </button>

          <div className="divider">
            <span>or use your email</span>
          </div>

          <form
            autoComplete="on"
            className="auth-form"
            onSubmit={handleEmailSubmit}
          >
            {mode === "register" && (
              <label className="auth-field">
                <span>Full Name</span>
                <input
                  autoComplete="name"
                  disabled={isPending}
                  name="name"
                  ref={displayNameInputRef}
                  required
                  type="text"
                  value={displayName}
                  onChange={(e) => setDisplayName(e.target.value)}
                />
              </label>
            )}

            <label className="auth-field">
              <span>Email</span>
              <input
                autoComplete={mode === "login" ? "username" : "email"}
                disabled={isPending}
                name={mode === "login" ? "username" : "email"}
                placeholder="you@example.com"
                ref={emailInputRef}
                required
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </label>


            <label className="auth-field auth-password-field">
              <span>Password</span>
              <div className="password-wrap">
                <input
                  autoComplete={
                    mode === "login" ? "current-password" : "new-password"
                  }
                  disabled={isPending}
                  minLength={6}
                  name={mode === "login" ? "password" : "new-password"}
                  placeholder="••••••••"
                  ref={passwordInputRef}
                  required
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                />
                <button
                  className="password-toggle"
                  disabled={isPending}
                  onClick={(e) => {
                    e.preventDefault();
                    setShowPassword((prev) => !prev);
                  }}
                  type="button"
                >
                  {showPassword ? "Hide" : "Show"}
                </button>
              </div>
            </label>
            {mode === "login" && (
              <button
                className="auth-link"
                onClick={async () => {
                  setError(null);
                  if (!emailInputRef.current?.value) {
                    setError(
                      "Enter your email first, then press Forgot password.",
                    );
                    return;
                  }
                  setIsPending(true);
                  try {
                    await sendPasswordResetEmail(
                      auth,
                      emailInputRef.current.value,
                    );
                    setError(
                      "If an account uses this email, a reset link is on its way. Check your inbox and spam folder.",
                    );
                  } catch {
                    setError(
                      "We could not send the reset email. Please try again.",
                    );
                  } finally {
                    setIsPending(false);
                  }
                }}
                disabled={isPending}
                type="button"
              >
                Forgot password?
              </button>
            )}

            {error && (
              <p className="auth-error" role="alert">
                {error}
              </p>
            )}

            <button
              className="primary-button auth-submit"
              disabled={isPending}
              type="submit"
            >
              {isPending
                ? "Please wait…"
                : mode === "login"
                  ? "Sign in"
                  : "Create account"}
            </button>
          </form>
          <p className="auth-legal">
            Not medical advice.{" "}
            <Link href="/privacy">How your data is handled</Link>
          </p>
        </div>
      </section>
    </main>
  );
}
