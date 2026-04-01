"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-context";

export default function LoginPage() {
  const { user, loading, signInWithGoogle, signInWithEmail, registerWithEmail } = useAuth();
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

  useEffect(() => {
    if (!loading && user) {
      router.replace("/dashboard");
    }
  }, [user, loading, router]);

  // Keep auth fields empty on mount in case browser autofill injects values.
  useEffect(() => {
    const clearAutofilledInputs = () => {
      if (displayNameInputRef.current?.value) {
        displayNameInputRef.current.value = "";
      }
      if (emailInputRef.current?.value) {
        emailInputRef.current.value = "";
      }
      if (passwordInputRef.current?.value) {
        passwordInputRef.current.value = "";
      }
      setDisplayName("");
      setEmail("");
      setPassword("");
    };

    clearAutofilledInputs();
    const timer = window.setTimeout(clearAutofilledInputs, 150);

    return () => {
      window.clearTimeout(timer);
    };
  }, []);

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
        await signInWithEmail(email, password);
      } else {
        if (!displayName.trim()) {
          setError("Please enter your full name.");
          setIsPending(false);
          return;
        }
        await registerWithEmail(email, password, displayName);
      }
      router.replace("/dashboard");
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Authentication failed.";
      // Surface friendly messages
      if (msg.includes("user-not-found") || msg.includes("wrong-password") || msg.includes("invalid-credential")) {
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
      router.replace("/dashboard");
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
      <main className="auth-shell">
        <div className="auth-loading">Loading…</div>
      </main>
    );
  }

  return (
    <main className="auth-shell">
      <section className="auth-visual" aria-hidden="true">
        <span className="auth-kicker">Clinical Intelligence Platform</span>
        <h2>See your health story with precision, not noise.</h2>
        <p>
          Upload medical reports, unify trends across labs, and get AI-powered context
          in one premium workspace built for clarity.
        </p>
        <div className="auth-highlights">
          <span>Trend-aware timelines</span>
          <span>Structured test extraction</span>
          <span>Contextual AI assistant</span>
        </div>
      </section>

      <section className="auth-panel">
        <div className="auth-card">
          <div className="auth-brand">
            <span className="auth-logo" aria-hidden="true">
              <svg viewBox="0 0 24 24" width="24" height="24" role="img" aria-label="Medical icon">
                <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.6" fill="none" />
                <path d="M5 12h3l1.3-2.3 2.2 5 1.9-3.2H19" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" fill="none" />
              </svg>
            </span>
            <h1>Medical Report Analyzer</h1>
            <p>Secure sign in to continue to your health workspace</p>
          </div>

          <div className="auth-tabs" role="tablist" aria-label="Authentication mode">
            <button
              className={`auth-tab ${mode === "login" ? "active" : ""}`}
              onClick={() => handleModeChange("login")}
              type="button"
              role="tab"
              aria-selected={mode === "login"}
            >
              Sign In
            </button>
            <button
              className={`auth-tab ${mode === "register" ? "active" : ""}`}
              onClick={() => handleModeChange("register")}
              type="button"
              role="tab"
              aria-selected={mode === "register"}
            >
              Create Account
            </button>
          </div>

          <button
            className="google-button"
            disabled={isPending}
            onClick={handleGoogle}
            type="button"
          >
            <span className="google-glyph" aria-hidden="true">G</span>
            Continue with Google
          </button>

          <div className="auth-divider"><span>or</span></div>

          <form autoComplete="off" className="auth-form" onSubmit={handleEmailSubmit}>
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

            {mode === "login" && (
              <button className="auth-link" onClick={(e) => e.preventDefault()} type="button">
                Forgot password?
              </button>
            )}

            <label className="auth-field auth-password-field">
              <span>Password</span>
              <div className="password-wrap">
                <input
                  autoComplete={mode === "login" ? "current-password" : "new-password"}
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

            {error && <p className="auth-error">{error}</p>}

            <button className="primary-button auth-submit" disabled={isPending} type="submit">
              {isPending
                ? "Please wait…"
                : mode === "login"
                ? "Sign In"
                : "Create Account"}
            </button>

            {mode === "login" ? (
              <p className="auth-footnote">
                Don&apos;t have an account?{" "}
                <button className="auth-inline-action" onClick={() => handleModeChange("register")} type="button">
                  Create one
                </button>
              </p>
            ) : (
              <p className="auth-footnote">
                Already have an account?{" "}
                <button className="auth-inline-action" onClick={() => handleModeChange("login")} type="button">
                  Sign in
                </button>
              </p>
            )}
          </form>
        </div>
      </section>
    </main>
  );
}
