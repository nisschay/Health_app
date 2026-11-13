"use client";
import { useRef, useState } from "react";
import { MoreHorizontal, Trash2, Pencil } from "lucide-react";
import { deleteStudy, renameStudy } from "@/lib/api";
export default function StudyActions({
  id,
  title,
  count,
  run,
  deleted,
  renamed,
}: {
  id: string;
  title: string;
  count: number;
  run: <T>(f: () => Promise<T>) => Promise<T>;
  deleted: () => Promise<void>;
  renamed: () => Promise<void>;
}) {
  const dialog = useRef<HTMLDialogElement>(null),
    renameDialog = useRef<HTMLDialogElement>(null),
    menu = useRef<HTMLDetailsElement>(null);
  const [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [name, setName] = useState(title);
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
            onClick={() => {
              if (menu.current) menu.current.open = false;
              setError("");
              setName(title);
              renameDialog.current?.showModal();
            }}
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
        ref={renameDialog}
        aria-labelledby="rename-study-title"
        onCancel={(e) => {
          if (busy) e.preventDefault();
        }}
      >
        <form
          onSubmit={async (e) => {
            e.preventDefault();
            const next = name.trim();
            if (!next || next === title) {
              renameDialog.current?.close();
              return;
            }
            setBusy(true);
            try {
              await run(() => renameStudy(id, next));
              await renamed();
              renameDialog.current?.close();
            } catch (err) {
              setError(
                err instanceof Error ? err.message : "Could not rename study.",
              );
            } finally {
              setBusy(false);
            }
          }}
        >
          <h2 id="rename-study-title">Rename study</h2>
          <label className="field rename-field">
            <span className="label">Name</span>
            <input
              className="input"
              value={name}
              maxLength={60}
              autoFocus
              required
              onChange={(e) => setName(e.target.value)}
            />
          </label>
          {error && <p role="alert">{error}</p>}
          <div className="actions">
            <button
              type="button"
              className="btn btn-secondary"
              disabled={busy}
              onClick={() => renameDialog.current?.close()}
            >
              Cancel
            </button>
            <button type="submit" className="btn btn-primary" disabled={busy}>
              {busy ? "Saving..." : "Save"}
            </button>
          </div>
        </form>
      </dialog>
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
