"use client";
import Brand from "../Brand";

import { useState } from "react";

type Props = {
  email: string | null;
  onContinue: () => Promise<boolean>;
  onResend: () => Promise<void>;
  onSignOut: () => void;
};

/** Shown instead of the dashboard until an email-and-password account confirms its address. */
export default function VerifyEmailPanel({
  email,
  onContinue,
  onResend,
  onSignOut,
}: Props) {
  const [busy, setBusy] = useState<"continue" | "resend" | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  async function run(kind: "continue" | "resend") {
    setBusy(kind);
    setMessage(null);
    try {
      if (kind === "resend") {
        await onResend();
        setMessage(
          "Sent. The link can take a minute to arrive; check spam too.",
        );
      } else if (!(await onContinue())) {
        setMessage(
          "This address is not confirmed yet. Open the link in the email, then press Continue.",
        );
      }
    } catch (err) {
      const text = err instanceof Error ? err.message : "";
      setMessage(
        text.includes("too-many-requests")
          ? "Too many emails sent. Wait a few minutes, then try again."
          : "Something went wrong. Please try again.",
      );
    } finally {
      setBusy(null);
    }
  }

  return (
    <main className="auth">
      <section className="auth-story">
        <h2 className="display">
          Follow your family&apos;s lab results over time.
        </h2>
        <p>
          Keep reports together. See what needs attention and what to ask your
          doctor.
        </p>
      </section>
      <div className="auth-form-side">
        <section className="auth-card" aria-labelledby="verify-email-heading">
          <div className="auth-brand brand">
            <Brand />
          </div>
          <h1 className="h1" id="verify-email-heading">
            Confirm your email
          </h1>
          <p>
            We sent a link to <strong>{email ?? "your email address"}</strong>.
            Open it to confirm the address is yours, then come back and press
            Continue. Your health reports stay locked to you, so we need to know
            the address works.
          </p>
          <div className="verify-email-actions">
            <button
              className="primary-button"
              type="button"
              disabled={busy !== null}
              onClick={() => {
                void run("continue");
              }}
            >
              {busy === "continue" ? "Checking…" : "I have confirmed, continue"}
            </button>
            <button
              className="secondary-button"
              type="button"
              disabled={busy !== null}
              onClick={() => {
                void run("resend");
              }}
            >
              {busy === "resend" ? "Sending…" : "Send the email again"}
            </button>
            <button className="link-button" type="button" onClick={onSignOut}>
              Sign out
            </button>
          </div>
          {message && (
            <p className="muted-copy" role="status">
              {message}
            </p>
          )}
        </section>
      </div>
    </main>
  );
}
