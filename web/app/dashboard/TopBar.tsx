"use client";
import { useState } from "react";
import { ChartNoAxesCombined, ChevronDown } from "lucide-react";
import { buildUserInitials } from "./helpers";
export function ThemePicker() {
  const [theme, setTheme] = useState(() =>
    typeof window === "undefined"
      ? "dark"
      : localStorage.getItem("health-app.theme") || "dark",
  );
  return (
    <label className="field">
      Theme
      <select
        className="input"
        value={theme}
        onChange={(e) => {
          const t = e.target.value;
          setTheme(t);
          localStorage.setItem("health-app.theme", t);
          if (t === "system")
            document.documentElement.dataset.theme = window.matchMedia(
              "(prefers-color-scheme: light)",
            ).matches
              ? "light"
              : "dark";
          else document.documentElement.dataset.theme = t;
        }}
      >
        {["system", "light", "dark"].map((t) => (
          <option key={t} value={t}>
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </option>
        ))}
      </select>
    </label>
  );
}
export default function TopBar({
  name,
  email,
  logout,
}: {
  name: string;
  email: string;
  logout: () => void;
}) {
  return (
    <header className="topbar">
      <div className="topbar-inner">
        <a className="brand" href="/dashboard">
          <span className="brand-mark">
            <ChartNoAxesCombined size={22} />
          </span>
          <span className="brand-name">LabTrail</span>
        </a>
        <span className="topbar-spacer" />
        <details
          className="account-menu"
          onKeyDown={(e) => {
            if (e.key === "Escape") {
              e.currentTarget.open = false;
              e.currentTarget.querySelector("summary")?.focus();
            }
          }}
        >
          <summary className="account-btn">
            <span className="avatar sm">{buildUserInitials(name, email)}</span>
            <span className="hide-sm">{name.split(" ")[0]}</span>
            <ChevronDown size={16} />
          </summary>
          <div className="card card-pad stack menu-content">
            <strong>{name}</strong>
            <p>{email}</p>
            <a href="/dashboard?view=account">Account and data</a>
            <ThemePicker />
            <a href="/privacy">Privacy</a>
            <button className="btn btn-secondary" onClick={logout}>
              Sign out
            </button>
          </div>
        </details>
      </div>
    </header>
  );
}
