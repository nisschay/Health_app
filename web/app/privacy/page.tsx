import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Privacy and data | LabTrail",
  description:
    "What the LabTrail stores, who processes it, and how to download or delete it.",
};

/** Plain statement of what happens to an uploaded report. Linked from sign-in and the upload form. */
export default function PrivacyPage() {
  return (
    <>
      <header className="topbar">
        <div className="topbar-inner">
          <Link className="brand" href="/dashboard">
            LabTrail
          </Link>
        </div>
      </header>
      <main className="legal-page">
        <Link href="/dashboard" className="legal-back">
          Back to the app
        </Link>
        <h1>Privacy and your data</h1>
        <p className="legal-updated">Last updated 12 November 2026</p>

        <h2>Not medical advice</h2>
        <p>
          This app reads lab reports and explains them in plain language. It can
          make mistakes, and it does not diagnose or treat anything. Check every
          result and every decision with a doctor.
        </p>

        <h2>What is stored</h2>
        <ul>
          <li>Your account: email, display name and sign-in times.</li>
          <li>
            The people you add (name, relationship, optional date of birth) and
            the studies you create.
          </li>
          <li>
            For each report: its file name, date and lab, and every test result
            read from it. The PDF itself is not kept after it is read.
          </li>
          <li>
            A cache of what the AI read from each of your reports, so
            re-uploading the same file is fast. It is yours alone.
          </li>
        </ul>

        <h2>Who processes it</h2>
        <ul>
          <li>
            <strong>Google Gemini</strong> reads the text of each uploaded
            report and answers assistant questions. This app uses the Gemini API
            free tier, under which Google may use what is sent to improve its
            products and humans may review it. Do not upload reports you are not
            comfortable sharing on those terms.
          </li>
          <li>
            <strong>Neon</strong> (Postgres) stores the data listed above.
          </li>
          <li>
            <strong>Firebase Authentication</strong> handles sign-in.
          </li>
          <li>
            <strong>Hugging Face</strong> runs the server that reads reports;{" "}
            <strong>Vercel</strong> serves this website.
          </li>
        </ul>

        <h2>Reports for family members</h2>
        <p>
          Upload someone else&apos;s report only with their permission, or as
          their carer.
        </p>

        <h2>Download or delete</h2>
        <p>
          On the dashboard, under <strong>Your data</strong>, you can download
          everything stored about you as one file, or delete your account.
          Deleting removes every report, result, study, profile, saved analysis
          and cached reading, and your sign-in, at once. It cannot be undone.
        </p>
      </main>
    </>
  );
}
