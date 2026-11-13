"use client";
import { useEffect, useState } from "react";
import {
  fetchProfiles,
  fetchStudiesForProfile,
  createProfile,
  createStudy,
  type ProfileItem,
  type StudySummary,
} from "@/lib/api";
import type { UploadJobState } from "./useUploadJob";
import { formatFileSize } from "./helpers";
import { FileText, Upload, X } from "lucide-react";
export function skipReason(s: string) {
  if (/No extractable text/i.test(s))
    return "No readable text. It looks like a photo or scan. Try the PDF the lab emailed you.";
  if (/Rate limit reached/i.test(s))
    return "Our reading service has reached its limit for today. Try this file again tomorrow.";
  if (/not a lab report|zero results/i.test(s))
    return "This does not look like a lab report, so nothing was read from it.";
  return s;
}
export default function UploadScreen({
  params,
  upload,
  run,
  reload,
}: {
  params: URLSearchParams;
  upload: UploadJobState;
  run: <T>(f: () => Promise<T>) => Promise<T>;
  reload: () => void;
}) {
  const [profiles, setProfiles] = useState<ProfileItem[]>([]);
  const [person, setPerson] = useState(params.get("profile") || "");
  const [studies, setStudies] = useState<StudySummary[]>([]);
  const [study, setStudy] = useState(params.get("study") || "");
  const [newPerson, setNewPerson] = useState(params.has("newPerson"));
  const [name, setName] = useState("");
  const [relation, setRelation] = useState("father");
  const [other, setOther] = useState("");
  const [dob, setDob] = useState("");
  const [studyName, setStudyName] = useState("");
  const [description, setDescription] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [uploadMode, setUploadMode] = useState<"new" | "append">("new");
  const [existingDataFile, setExistingDataFile] = useState<File | null>(null);
  const [consent, setConsent] = useState(false);
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);
  useEffect(() => {
    void run(fetchProfiles)
      .then((p) => {
        setProfiles(p);
        if (p.length === 1 && !params.has("newPerson")) setPerson(p[0]!.id);
      })
      .catch((e) => setError(e.message));
  }, [run, params]);
  useEffect(() => {
    if (!person) return;
    let active = true;
    void run(() => fetchStudiesForProfile(person))
      .then((s) => {
        if (!active) return;
        setStudies(s);
        setStudy(
          s.some((x) => x.id === params.get("study"))
            ? params.get("study")!
            : s.length === 1
              ? s[0]!.id
              : "",
        );
      })
      .catch((e) => setError(e.message));
    return () => {
      active = false;
    };
  }, [person, run, params]);
  const selected = profiles.find((p) => p.id === person);
  const suggested = `${selected?.full_name.split(" ")[0] || "My"}'s reports ${new Date().getFullYear()}`;
  function addFiles(incoming: File[]) {
    const next = [...files];
    const errors: string[] = [];
    for (const f of incoming) {
      if (!/\.pdf$/i.test(f.name)) {
        errors.push(`${f.name} is not a PDF, so it was left out.`);
        continue;
      }
      if (
        f.size > 10 * 1024 * 1024 ||
        next.length >= 20 ||
        next.reduce((n, x) => n + x.size, 0) + f.size > 25 * 1024 * 1024
      ) {
        errors.push(`${f.name} exceeds the file limits, so it was left out.`);
        continue;
      }
      if (!next.some((x) => x.name === f.name)) next.push(f);
    }
    setFiles(next);
    setError(errors.join(" "));
  }
  async function submit() {
    if (pending || upload.busy) return;
    setPending(true);
    setError("");
    try {
      let id = study;
      if (!id) {
        const s = await run(() =>
          createStudy({
            profile_id: person,
            name: studyName.trim() || suggested,
            description,
          }),
        );
        id = s.id;
        setStudy(id);
        setStudies((prev) => [...prev, s]);
      }
      await upload.start(
        files,
        id,
        uploadMode === "append" ? existingDataFile : null,
      );
      reload();
    } catch (e) {
      setError(
        e instanceof Error ? e.message : "Could not create study. Try again.",
      );
    } finally {
      setPending(false);
    }
  }
  const missing = !person
    ? "Choose who these reports are for"
    : !files.length && !(uploadMode === "append" && existingDataFile)
      ? "Choose at least one PDF or spreadsheet"
      : !consent
        ? "Confirm that you have permission to upload these reports"
        : "";
  const j = upload.job;
  const finished = j?.status === "done";
  return (
    <main className="page page-narrow stack">
      <nav className="crumbs">
        <a href="/dashboard">← Family</a>
      </nav>
      <h1 className="display">
        {finished
          ? `${Object.values(j.progress.files).filter((f) => f.step === "done").length} reports added`
          : upload.busy
            ? `Reading ${j?.progress.total ?? files.length} reports`
            : upload.error
              ? "We could not read these reports"
              : "Add reports"}
      </h1>
      {selected && (
        <p className="soft">
          For {selected.full_name}
          {study &&
            ` · ${studies.find((s) => s.id === study)?.name || "your study"}`}
        </p>
      )}
      {(error || upload.error) && (
        <div className="banner banner-critical" role="alert">
          {error || upload.error}
        </div>
      )}
      {upload.busy || finished ? (
        <section className="card card-pad stack" aria-live="polite">
          {!finished && (
            <>
              <p>
                {j?.progress.processed ?? 0} of{" "}
                {j?.progress.total ?? files.length} finished
              </p>
              <progress
                max={j?.progress.total || files.length || 1}
                value={j?.progress.processed ?? 0}
              />
              {j?.progress.eta_seconds != null && (
                <p>
                  {j.progress.eta_seconds < 60
                    ? "Less than a minute left"
                    : `About ${Math.ceil(j.progress.eta_seconds / 60)} minutes left`}
                </p>
              )}
            </>
          )}
          <ul className="file-list">
            {Object.entries(
              j?.progress.files ??
                Object.fromEntries(
                  files.map((f) => [f.name, { step: "queued", percent: 0 }]),
                ),
            ).map(([file, f]) => (
              <li className="file-row" key={file}>
                <FileText size={20} />
                <div>
                  <span className="file-name" title={file}>
                    {file}
                  </span>
                  {"error" in f && typeof f.error === "string" && f.error && (
                    <p className="file-sub err">{skipReason(f.error)}</p>
                  )}
                </div>
                <span className="file-state">
                  {!j
                    ? "Uploading"
                    : (
                        {
                          queued: "Waiting",
                          extracting: "Reading the report",
                          parsing: "Sorting results",
                          done: "Done",
                          failed: "Skipped",
                        } as Record<string, string>
                      )[f.step]}
                </span>
              </li>
            ))}
          </ul>
          {!!j?.progress.notes?.length && (
            <div>
              <h2 className="h2">
                Notes about these reports
              </h2>
              {j.progress.notes.map((n) => (
                <p key={n}>{n}</p>
              ))}
            </div>
          )}
          {finished ? (
            <>
              <a
                className="btn btn-primary"
                href={
                  j.study_id
                    ? `?study=${j.study_id}`
                    : `?analysis=${j.analysis_id}`
                }
              >
                See the results
              </a>
              <button
                className="btn btn-secondary"
                onClick={() => {
                  upload.clear();
                  setFiles([]);
                }}
              >
                Add other files
              </button>
            </>
          ) : (
            <p>
              You can close this page or lock your phone. We keep reading and
              save the results to your account.
            </p>
          )}
        </section>
      ) : (
        <form
          className="card"
          onSubmit={(e) => {
            e.preventDefault();
            void submit();
          }}
        >
          <section className="step">
            <span className="step-num hide-sm">1</span>
            <div className="stack">
              <h2 className="h2">Who are these reports for?</h2>
              <div className="choices">
                {profiles.map((p) => (
                  <button
                    className="choice"
                    type="button"
                    key={p.id}
                    aria-pressed={person === p.id}
                    onClick={() => {
                      setPerson(p.id);
                      setStudy("");
                      setStudyName("");
                      setNewPerson(false);
                    }}
                  >
                    {p.full_name}
                    <span className="choice-sub">
                      {p.relationship === "self" ? "You" : p.relationship}
                    </span>
                  </button>
                ))}
                <button
                  type="button"
                  className="choice dashed"
                  onClick={() => setNewPerson(true)}
                >
                  Someone new
                </button>
              </div>
              {newPerson && (
                <div className="stack">
                  <label className="field">
                    Full name
                    <input
                      className="input"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                    />
                  </label>
                  <label className="field">
                    Relationship
                    <select
                      className="input"
                      value={relation}
                      onChange={(e) => setRelation(e.target.value)}
                    >
                      {[
                        "father",
                        "mother",
                        "spouse",
                        "son",
                        "daughter",
                        "brother",
                        "sister",
                        "grandparent",
                        "other",
                      ].map((r) => (
                        <option key={r}>{r}</option>
                      ))}
                    </select>
                  </label>
                  {relation === "other" && (
                    <label className="field">
                      Relationship name
                      <input
                        className="input"
                        value={other}
                        onChange={(e) => setOther(e.target.value)}
                      />
                    </label>
                  )}
                  <label className="field">
                    Date of birth (optional)
                    <input
                      className="input"
                      type="date"
                      value={dob}
                      onChange={(e) => setDob(e.target.value)}
                    />
                  </label>
                  <button
                    type="button"
                    className="btn btn-secondary"
                    disabled={
                      !name.trim() ||
                      pending ||
                      (relation === "other" && !other.trim())
                    }
                    onClick={async () => {
                      setPending(true);
                      try {
                        const p = await run(() =>
                          createProfile({
                            full_name: name.trim(),
                            relationship:
                              relation === "other" ? other.trim() : relation,
                            date_of_birth: dob || null,
                          }),
                        );
                        setProfiles((prev) => [...prev, p]);
                        setPerson(p.id);
                        setNewPerson(false);
                        reload();
                      } catch (e) {
                        setError(
                          e instanceof Error
                            ? e.message
                            : "Could not add person.",
                        );
                      } finally {
                        setPending(false);
                      }
                    }}
                  >
                    Add person
                  </button>
                  <button
                    className="btn btn-quiet"
                    type="button"
                    onClick={() => setNewPerson(false)}
                  >
                    Cancel
                  </button>
                </div>
              )}
            </div>
          </section>
          <section className="step">
            <span className="step-num hide-sm">2</span>
            <div className="stack">
              <h2 className="h2">Which study do they belong to?</h2>
              {!person ? (
                <p>Choose a person first</p>
              ) : (
                <>
                  <div className="choices">
                    {studies.map((s) => (
                      <button
                        className="choice"
                        type="button"
                        key={s.id}
                        aria-pressed={study === s.id}
                        onClick={() => setStudy(s.id)}
                      >
                        {s.name} · {s.report_count} reports
                      </button>
                    ))}
                    <button
                      type="button"
                      className="choice"
                      aria-pressed={!study}
                      onClick={() => setStudy("")}
                    >
                      New study
                    </button>
                  </div>
                  {!study && (
                    <>
                      <label className="field">
                        Study name
                        <input
                          className="input"
                          maxLength={60}
                          required
                          value={studyName || suggested}
                          onChange={(e) => setStudyName(e.target.value)}
                        />
                      </label>
                      <details>
                        <summary>Add a note</summary>
                        <label className="field">
                          Note (optional)
                          <textarea
                            className="textarea"
                            maxLength={200}
                            value={description}
                            onChange={(e) => setDescription(e.target.value)}
                          />
                        </label>
                      </details>
                    </>
                  )}
                </>
              )}
              <p className="small muted">
                A study keeps related reports together, so each test can be
                compared over time.
              </p>
            </div>
          </section>
          <section className="step">
            <span className="step-num hide-sm">3</span>
            <div className="stack">
              <h2 className="h2">Choose the PDF files</h2>
              <label
                className="dropzone"
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  addFiles(Array.from(e.dataTransfer.files));
                }}
              >
                <Upload />
                <strong>Drop PDF reports here, or choose files</strong>
                <span className="small muted">
                  Up to 20 files · 10 MB each · 25 MB in total
                </span>
                <input
                  className="input"
                  type="file"
                  accept="application/pdf,.pdf"
                  multiple
                  onChange={(e) => {
                    addFiles(Array.from(e.target.files ?? []));
                    e.target.value = "";
                  }}
                />
              </label>
              <ul className="file-list">
                {files.map((f) => (
                  <li className="file-row" key={f.name}>
                    <FileText size={20} />
                    <div>
                      <span className="file-name">{f.name}</span>
                      <span className="file-sub">{formatFileSize(f.size)}</span>
                    </div>
                    <button
                      className="icon-btn"
                      type="button"
                      aria-label={`Remove ${f.name}`}
                      onClick={() => setFiles(files.filter((x) => x !== f))}
                    >
                      <X size={18} />
                    </button>
                  </li>
                ))}
              </ul>
              <details open={!!existingDataFile || undefined}>
                <summary>Already have a spreadsheet of results? Add it</summary>
                <label className="field">
                  Excel or CSV results
                  <input
                    className="input"
                    type="file"
                    accept=".xlsx,.csv"
                    onChange={(e) => {
                      // The spreadsheet itself decides the mode, so closing this section cannot drop it silently.
                      const sheet = e.target.files?.[0] ?? null;
                      setExistingDataFile(sheet);
                      setUploadMode(sheet ? "append" : "new");
                    }}
                  />
                </label>
                <p className="small muted">
                  Add to Existing Data. Choose one spreadsheet alongside or
                  instead of PDFs.
                </p>
              </details>
              <label className="check">
                <input
                  type="checkbox"
                  checked={consent}
                  onChange={(e) => setConsent(e.target.checked)}
                />
                <span>
                  I have permission to upload these reports. I understand their
                  text is sent to an AI service (Groq or Google Gemini) to be read, and that
                  the results are not medical advice.{" "}
                  <a href="/privacy">How your data is handled</a>
                </span>
              </label>
              <button
                className="btn btn-primary btn-block"
                disabled={!!missing || pending}
              >
                {pending
                  ? "Preparing..."
                  : files.length
                    ? `Read ${files.length} ${files.length === 1 ? "report" : "reports"}${existingDataFile ? " and 1 spreadsheet" : ""}`
                    : existingDataFile
                      ? "Read spreadsheet results"
                      : "Read reports"}
              </button>
              {missing && <p className="small muted">{missing}</p>}
            </div>
          </section>
        </form>
      )}
      <p className="disclaimer">
        The PDF itself is not kept after it is read. Only the results are
        stored. <a href="/privacy">How your data is handled</a>
      </p>
    </main>
  );
}
