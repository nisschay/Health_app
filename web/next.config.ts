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

if (process.env.NODE_ENV === "development" && process.env.VERCEL !== "1") {
  loadCentralEnv();
}

const nextConfig: NextConfig = {
  // Keep dev and production artifacts isolated so running `next build`
  // does not invalidate chunks served by `next dev`.
  distDir: process.env.NODE_ENV === "development" ? ".next-dev" : ".next",
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