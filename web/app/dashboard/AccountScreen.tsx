"use client";

import { useAuth } from "@/lib/auth-context";
import { ThemePicker } from "./TopBar";
import { useState } from "react";
import { deleteUser } from "firebase/auth";
import { auth } from "@/lib/firebase";
import { deleteMyAccount, downloadMyData } from "@/lib/api";

type Props = {
  runWithTokenRetry: <T>(operation: () => Promise<T>) => Promise<T>;
  onAccountDeleted: () => void;
};

const CONFIRM_WORD = "DELETE";

/** Download everything stored about you, or delete it. */
export default function AccountScreen({
  runWithTokenRetry,
  onAccountDeleted,
}: Props) {
  const { user, logout } = useAuth();
  const [busy, setBusy] = useState<"export" | "delete" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirming, setConfirming] = useState(false);
  const [typed, setTyped] = useState("");

  async function handleDownload() {
    setBusy("export");
    setError(null);
    try {
      const blob = await runWithTokenRetry(() => downloadMyData());
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "my-health-data.json";
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Could not prepare the download.",
      );
    } finally {
      setBusy(null);
    }
  }

  async function handleDelete() {
    setBusy("delete");
    setError(null);
    try {
      await runWithTokenRetry(() => deleteMyAccount());
      // The server removed the data and the login. This is only a fallback if it could not remove the login.
      if (auth.currentUser)
        await deleteUser(auth.currentUser).catch(() => undefined);
      onAccountDeleted();
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not delete your data. Nothing was removed.",
      );
      setBusy(null);
    }
  }

  return (
    <section
      className="card card-pad stack"
      aria-labelledby="your-data-heading"
    >
      <h2 id="your-data-heading">Account and data</h2>
      <p className="muted-copy">
        Download every profile, report, result and saved analysis as one file,
        or delete all of it.
      </p>
      <section className="stack">
        <h3 className="h2">You</h3>
        <p>{user?.displayName}</p>
        <p>{user?.email}</p>
        <p>
          Signed in with{" "}
          {user?.providerData.some((p) => p.providerId === "google.com")
            ? "Google"
            : "email"}
        </p>
        <button className="btn btn-secondary" onClick={() => void logout()}>
          Sign out
        </button>
      </section>
      <section className="stack">
        <h3 className="h2">Appearance</h3>
        <ThemePicker />
      </section>
      <h3 className="h2">Your data</h3>
      <div className="row row-wrap">
        <button
          className="secondary-button"
          type="button"
          disabled={busy !== null}
          onClick={() => void handleDownload()}
        >
          {busy === "export" ? "Preparing…" : "Download my data"}
        </button>
        {!confirming && (
          <button
            className="danger-button"
            type="button"
            disabled={busy !== null}
            onClick={() => setConfirming(true)}
          >
            Delete my account
          </button>
        )}
      </div>
      {confirming && (
        <div className="danger-confirm">
          <label htmlFor="confirm-delete">
            This permanently deletes every report, result, study, profile and
            saved analysis. Type {CONFIRM_WORD} to confirm.
          </label>
          <div className="danger-confirm-row">
            <input
              id="confirm-delete"
              value={typed}
              onChange={(e) => setTyped(e.target.value)}
              autoComplete="off"
            />
            <button
              className="danger-button"
              type="button"
              disabled={typed !== CONFIRM_WORD || busy !== null}
              onClick={() => void handleDelete()}
            >
              {busy === "delete" ? "Deleting…" : "Delete everything"}
            </button>
            <button
              className="secondary-button"
              type="button"
              disabled={busy !== null}
              onClick={() => {
                setConfirming(false);
                setTyped("");
              }}
            >
              Cancel
            </button>
          </div>
        </div>
      )}
      {error && (
        <p className="error-text" role="alert">
          {error}
        </p>
      )}
    </section>
  );
}
