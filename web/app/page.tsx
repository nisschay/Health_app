const featureCards = [
  {
    title: "Upload and Analyze",
    body: "Patients upload one or more PDF reports and get structured extraction, normalized lab data, and a reusable patient timeline."
  },
  {
    title: "Clinical Trend Views",
    body: "The new product direction keeps the current trend analysis but presents it in a cleaner patient-facing dashboard for desktop and mobile."
  },
  {
    title: "Authenticated Report History",
    body: "Firebase-backed users, storage, and saved report history replace the current in-memory Streamlit session model."
  }
];

const milestones = [
  "FastAPI layer wrapping the existing Python medical pipeline",
  "Next.js frontend designed for Vercel deployment",
  "Firebase-ready auth contract for real users",
  "Future phases for RAG, agents, and fine-tuning"
];

export default function HomePage() {
  return (
    <main className="shell">
      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Medical analytics migration in progress</p>
          <h1>Medical reports, translated into a product patients can actually use.</h1>
          <p className="lede">
            This frontend is the new Vercel-ready shell for the Medical Project: a cleaner,
            more scalable replacement for the current Streamlit experience while keeping the
            Python analysis core that already works.
          </p>

          <div className="actions">
            <a className="primary-button" href="/dashboard">
              Open Dashboard Prototype
            </a>
            <a className="secondary-button" href="http://localhost:8000/docs">
              View Backend API Docs
            </a>
          </div>
        </div>

        <div className="hero-panel">
          <div className="panel-card accent-card">
            <span className="panel-label">Current backend slice</span>
            <strong>FastAPI + Gemini + existing Python logic</strong>
            <p>
              The backend already exposes analysis, chat, insights, and PDF export routes so
              the frontend can move independently from Streamlit.
            </p>
          </div>

          <div className="panel-grid">
            <div className="panel-card stat-card">
              <span className="panel-label">Status</span>
              <strong>Migration started</strong>
            </div>
            <div className="panel-card stat-card">
              <span className="panel-label">Users</span>
              <strong>Firebase-ready</strong>
            </div>
            <div className="panel-card stat-card wide-card">
              <span className="panel-label">Frontend goal</span>
              <strong>Responsive dashboard on Vercel</strong>
            </div>
          </div>
        </div>
      </section>

      <section className="section-grid">
        {featureCards.map((card) => (
          <article className="feature-card" key={card.title}>
            <h2>{card.title}</h2>
            <p>{card.body}</p>
          </article>
        ))}
      </section>

      <section className="roadmap-strip">
        <div>
          <p className="eyebrow">Delivery path</p>
          <h2>What this branch already changes</h2>
        </div>
        <ol className="milestone-list">
          {milestones.map((milestone) => (
            <li key={milestone}>{milestone}</li>
          ))}
        </ol>
      </section>
    </main>
  );
}