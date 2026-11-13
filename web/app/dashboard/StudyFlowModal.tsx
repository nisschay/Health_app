"use client";

import { useEffect } from "react";
import type { ProfileItem, StudySummary } from "@/lib/api";
import { fmtRange, fmtRelativeDate, relationshipLabel } from "./helpers";
import type { StudyAction } from "./types";

export default function StudyFlowModal({
  open,
  userName,
  profiles,
  selectedProfile,
  studies,
  selectedAction,
  selectedStudy,
  step,
  loading,
  formError,
  onClose,
  onSelectProfile,
  onSelectAction,
  onSelectStudy,
  onConfirmAdd,
  onBack,
  newStudyName,
  newStudyDescription,
  onChangeNewStudyName,
  onChangeNewStudyDescription,
  onCreateStudy,
  newMemberName,
  newMemberRelationship,
  newMemberOtherRelationship,
  newMemberDob,
  onChangeMemberName,
  onChangeMemberRelationship,
  onChangeMemberOtherRelationship,
  onChangeMemberDob,
  onCreateMember,
  onStartNewMember,
}: {
  open: boolean;
  userName: string;
  profiles: ProfileItem[];
  selectedProfile: ProfileItem | null;
  studies: StudySummary[];
  selectedAction: StudyAction | null;
  selectedStudy: StudySummary | null;
  step: "who" | "what" | "pick-study" | "confirm-add" | "new-study" | "new-member";
  loading: boolean;
  formError: string | null;
  onClose: () => void;
  onSelectProfile: (profile: ProfileItem) => void;
  onSelectAction: (action: StudyAction) => void;
  onSelectStudy: (study: StudySummary) => void;
  onConfirmAdd: () => void;
  onBack: () => void;
  newStudyName: string;
  newStudyDescription: string;
  onChangeNewStudyName: (value: string) => void;
  onChangeNewStudyDescription: (value: string) => void;
  onCreateStudy: () => void;
  newMemberName: string;
  newMemberRelationship: string;
  newMemberOtherRelationship: string;
  newMemberDob: string;
  onChangeMemberName: (value: string) => void;
  onChangeMemberRelationship: (value: string) => void;
  onChangeMemberOtherRelationship: (value: string) => void;
  onChangeMemberDob: (value: string) => void;
  onCreateMember: () => void;
  onStartNewMember: () => void;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => { if (event.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  const sortedProfiles = [...profiles].sort((a, b) => {
    const aSelf = a.relationship.toLowerCase() === "self" ? 0 : 1;
    const bSelf = b.relationship.toLowerCase() === "self" ? 0 : 1;
    if (aSelf !== bSelf) return aSelf - bSelf;
    return a.full_name.localeCompare(b.full_name);
  });

  const stepMeta: Record<typeof step, { index: number; total: number }> = {
    who: { index: 1, total: 3 },
    what: { index: 2, total: 3 },
    "pick-study": { index: 3, total: 4 },
    "confirm-add": { index: 4, total: 4 },
    "new-study": { index: 3, total: 3 },
    "new-member": { index: 2, total: 3 },
  };
  const { index, total } = stepMeta[step];

  return (
    <div className="study-flow-backdrop" role="dialog" aria-modal="true" aria-label="Choose where to save these reports">
      <div className="study-flow-modal">
        <div className="study-flow-head">
          <div className="study-flow-progress" aria-label="Step indicator">
            <span>Step {index} of {total}</span>
            <div className="study-flow-dots">
              {Array.from({ length: total }).map((_, idx) => (
                <i key={idx} className={idx + 1 <= index ? "active" : ""} />
              ))}
            </div>
          </div>
          <button className="secondary-button" type="button" onClick={onClose}>Close</button>
        </div>

        {step === "who" && (
          <section className="study-step-panel">
            <h3>Who are these reports for?</h3>
            <p>Select a profile first, then choose whether to append or start a fresh study.</p>
            <div className="study-card-grid">
              {sortedProfiles.map((profile) => {
                const isSelf = profile.relationship.toLowerCase() === "self";
                const label = isSelf
                  ? `Myself - ${userName || profile.full_name}`
                  : `${profile.full_name} - ${relationshipLabel(profile.relationship)}`;
                return (
                  <button
                    key={profile.id}
                    className={`study-choice-card ${selectedProfile?.id === profile.id ? "active" : ""}`}
                    type="button"
                    onClick={() => onSelectProfile(profile)}
                    disabled={loading}
                  >
                    <strong>{label}</strong>
                    <span>{isSelf ? "Primary account profile" : "Family profile"}</span>
                  </button>
                );
              })}
              <button className="study-choice-card add" type="button" onClick={onStartNewMember} disabled={loading}>
                <strong>Add a New Family Member</strong>
                <span>Create profile and continue to first study setup</span>
              </button>
            </div>
            {sortedProfiles.length === 0 && loading && <p className="muted-copy">Loading your profiles…</p>}
            {sortedProfiles.length === 0 && !loading && (
              <p className="muted-copy">
                We could not load your existing profiles yet. Try closing this dialog and reopening it after your session refreshes.
              </p>
            )}
          </section>
        )}

        {step === "what" && selectedProfile && (
          <section className="study-step-panel">
            <h3>What would you like to do for {selectedProfile.full_name}?</h3>
            <div className="study-card-grid two-col">
              <button
                className={`study-choice-card ${selectedAction === "add-existing" ? "active" : ""}`}
                type="button"
                onClick={() => onSelectAction("add-existing")}
                disabled={loading}
              >
                <strong>Add to an existing study</strong>
                <span>Upload new reports and merge trends into an existing study.</span>
              </button>
              <button
                className={`study-choice-card ${selectedAction === "start-new" ? "active" : ""}`}
                type="button"
                onClick={() => onSelectAction("start-new")}
                disabled={loading}
              >
                <strong>Start a new study</strong>
                <span>Begin an independent study for this same profile.</span>
              </button>
            </div>
            <div className="study-step-actions">
              <button className="secondary-button" type="button" onClick={onBack}>Back</button>
            </div>
          </section>
        )}

        {step === "pick-study" && selectedProfile && (
          <section className="study-step-panel">
            <h3>Select an existing study for {selectedProfile.full_name}</h3>
            <div className="study-list-stack">
              {studies.map((study) => (
                <button
                  key={study.id}
                  className={`study-list-card ${selectedStudy?.id === study.id ? "active" : ""}`}
                  type="button"
                  onClick={() => onSelectStudy(study)}
                >
                  <strong>{study.name}</strong>
                  <p>{study.report_count} reports • {fmtRange(study.range_start, study.range_end)}</p>
                  <span>Last updated {fmtRelativeDate(study.last_updated)}</span>
                </button>
              ))}
            </div>
            <div className="study-step-actions">
              <button className="secondary-button" type="button" onClick={onBack}>Back</button>
            </div>
          </section>
        )}

        {step === "confirm-add" && selectedStudy && (
          <section className="study-step-panel">
            <h3>Add to {selectedStudy.name}?</h3>
            <p>
              {selectedStudy.report_count > 0
                ? `New reports will be analyzed together with the ${selectedStudy.report_count} already in this study, so trends cover all of them.`
                : "This study has no reports yet; these will be its first."}
            </p>
            <div className="study-step-actions">
              <button className="primary-button" type="button" onClick={onConfirmAdd}>Yes, Add to This Study</button>
              <button className="secondary-button" type="button" onClick={onBack}>Go Back</button>
            </div>
          </section>
        )}

        {step === "new-study" && selectedProfile && (
          <section className="study-step-panel">
            <h3>Create a new study for {selectedProfile.full_name}</h3>
            <label className="field-block">
              <span>Study Name</span>
              <input
                className="input"
                maxLength={60}
                placeholder="e.g. Cardiac Monitoring 2025"
                value={newStudyName}
                onChange={(e) => onChangeNewStudyName(e.target.value)}
              />
            </label>
            <label className="field-block">
              <span>Description (optional)</span>
              <textarea
                className="textarea"
                maxLength={200}
                placeholder="What is this study tracking?"
                value={newStudyDescription}
                onChange={(e) => onChangeNewStudyDescription(e.target.value)}
              />
            </label>
            <div className="study-step-actions">
              <button className="primary-button" type="button" onClick={onCreateStudy} disabled={loading}>
                Create Study and Upload Reports
              </button>
              <button className="secondary-button" type="button" onClick={onBack}>Back</button>
            </div>
          </section>
        )}

        {step === "new-member" && (
          <section className="study-step-panel">
            <h3>Add a new family member</h3>
            <label className="field-block">
              <span>Full Name</span>
              <input
                className="input"
                placeholder="Enter their full name"
                value={newMemberName}
                onChange={(e) => onChangeMemberName(e.target.value)}
              />
            </label>
            <label className="field-block">
              <span>Relationship</span>
              <select
                className="select-input"
                value={newMemberRelationship}
                onChange={(e) => onChangeMemberRelationship(e.target.value)}
              >
                <option value="father">Father</option>
                <option value="mother">Mother</option>
                <option value="spouse">Spouse</option>
                <option value="child">Child</option>
                <option value="sibling">Sibling</option>
                <option value="grandparent">Grandparent</option>
                <option value="other">Other</option>
              </select>
            </label>
            {newMemberRelationship === "other" && (
              <label className="field-block">
                <span>Specify Relationship</span>
                <input
                  className="input"
                  placeholder="e.g. Cousin"
                  value={newMemberOtherRelationship}
                  onChange={(e) => onChangeMemberOtherRelationship(e.target.value)}
                />
              </label>
            )}
            <label className="field-block">
              <span>Date of Birth (optional)</span>
              <input
                className="text-input"
                type="date"
                value={newMemberDob}
                max={new Date().toISOString().slice(0, 10)}
                onChange={(e) => onChangeMemberDob(e.target.value)}
              />
            </label>
            <small className="muted-copy">Helps with age-specific reference ranges in analysis.</small>
            <div className="study-step-actions">
              <button className="primary-button" type="button" onClick={onCreateMember} disabled={loading}>
                Create Profile and Continue
              </button>
              <button className="secondary-button" type="button" onClick={onBack}>Back</button>
            </div>
          </section>
        )}

        {formError && <p className="status-text error-text">{formError}</p>}
      </div>
    </div>
  );
}
