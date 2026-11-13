"use client";
import { useRef, useState } from "react";
import { MoreHorizontal, Trash2, Pencil } from "lucide-react";
import { deleteStudy } from "@/lib/api";
export default function StudyActions({
  id,
  title,
  count,
  run,
  deleted,
}: {
  id: string;
  title: string;
  count: number;
  run: <T>(f: () => Promise<T>) => Promise<T>;
  deleted: () => Promise<void>;
}) {
  const dialog = useRef<HTMLDialogElement>(null),
    menu = useRef<HTMLDetailsElement>(null);
  const [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  return (
    <>
      <details
        className="study-actions"
        ref={menu}
        onKeyDown={(e) => {
          if (e.key === "Escape") {
            e.currentTarget.open = false;
            e.currentTarget.querySelector("summary")?.focus();
          }
        }}
      >
        <summary className="icon-btn" aria-label="Study actions">
          <MoreHorizontal size={18} />
        </summary>
        <div className="study-menu">
          <button
            disabled
            title="Renaming studies is not supported by the server yet"
          >
            <Pencil size={16} />
            Rename study
          </button>
          <hr />
          <button
            className="danger"
            onClick={() => {
              if (menu.current) menu.current.open = false;
              setError("");
              dialog.current?.showModal();
            }}
          >
            <Trash2 size={16} />
            Delete study
          </button>
        </div>
      </details>
      <dialog
        className="confirm-dialog"
        ref={dialog}
        aria-labelledby="delete-study-title"
        onCancel={(e) => {
          if (busy) e.preventDefault();
        }}
      >
        <h2 id="delete-study-title">Delete {title}?</h2>
        <p>
          This removes this study, its {count} reports and all results in those
          reports. Your profile and other studies stay. This cannot be undone.
        </p>
        {error && <p role="alert">{error}</p>}
        <div className="actions">
          <button
            className="btn btn-secondary"
            autoFocus
            disabled={busy}
            onClick={() => dialog.current?.close()}
          >
            Cancel
          </button>
          <button
            className="btn btn-danger"
            disabled={busy}
            onClick={async () => {
              setBusy(true);
              try {
                await run(() => deleteStudy(id));
                dialog.current?.close();
                await deleted();
              } catch (e) {
                setError(
                  e instanceof Error ? e.message : "Could not delete study.",
                );
              } finally {
                setBusy(false);
              }
            }}
          >
            {busy ? "Deleting..." : "Delete study"}
          </button>
        </div>
      </dialog>
    </>
  );
}
