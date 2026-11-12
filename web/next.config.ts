import type { NextConfig } from "next";
import fs from "fs";
import path from "path";

function loadCentralEnv() {
  const envPath = path.resolve(process.cwd(), "..", ".env");
  if (!fs.existsSync(envPath)) return;

  const content = fs.readFileSync(envPath, "utf8");
  for (const rawLine of content.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const equalsAt = line.indexOf("=");
    if (equalsAt <= 0) continue;

    const key = line.slice(0, equalsAt).trim();
    let value = line.slice(equalsAt + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"'))
      || (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }

    // Never clobber env values already injected by CI/hosting providers.
    if (process.env[key] === undefined) {
      process.env[key] = value;
    }
  }
}

// Only load central env in local development-like environments.
if (process.env.NODE_ENV !== "production" && process.env.VERCEL !== "1") {
  loadCentralEnv();
}

// Previews may build without Firebase settings, but a production site without them
// cannot sign anyone in. Fail that build instead of shipping a broken login.
const REQUIRED_IN_PRODUCTION = [
  "NEXT_PUBLIC_FIREBASE_API_KEY",
  "NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN",
  "NEXT_PUBLIC_FIREBASE_PROJECT_ID",
  "NEXT_PUBLIC_FIREBASE_APP_ID",
];
if (process.env.VERCEL_ENV === "production") {
  const missing = REQUIRED_IN_PRODUCTION.filter((key) => !process.env[key]);
  if (missing.length) throw new Error(`Missing production environment variables: ${missing.join(", ")}`);
}

const nextConfig: NextConfig = {
  // Keep dev and production artifacts isolated so running `next build`
  // does not invalidate chunks served by `next dev`.
  distDir: process.env.NODE_ENV === "development" ? ".next-dev" : ".next",
  compiler: {
    // Report content was reaching the browser console in production.
    removeConsole: process.env.NODE_ENV === "production" ? { exclude: ["error"] } : false,
  },
  experimental: {
    optimizePackageImports: ["firebase"],
    // Keep client upload limits generous for local proxy/dev flows.
    middlewareClientMaxBodySize: "300mb",
    // Analyze can take several minutes for large report batches.
    proxyTimeout: 15 * 60 * 1000,
  },
  async rewrites() {
    // Production should call backend directly via NEXT_PUBLIC_API_URL.
    if (process.env.VERCEL === "1") return [];
    return [
      {
        source: "/backend/:path*",
        destination: "http://127.0.0.1:8000/:path*",
      },
    ];
  },
};

export default nextConfig;