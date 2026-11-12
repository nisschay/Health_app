"use client";

/** The bar across the top of the dashboard. */

export default function NavBar({ user, onLogout, onHome }: { user: { displayName?: string | null; email?: string | null } | null; onLogout: () => void; onHome: () => void }) {
  return (
    <nav className="top-nav">
      <button className="nav-brand" onClick={onHome} type="button">Medical Report Analyzer</button>
      <div className="nav-right">
        {user && <span className="nav-user">{user.displayName ?? user.email ?? "User"}</span>}
        <button className="secondary-button nav-logout" onClick={onLogout} type="button">Sign Out</button>
      </div>
    </nav>
  );
}
