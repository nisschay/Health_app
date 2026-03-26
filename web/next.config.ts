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

    process.env[key] = value;
  }
}

loadCentralEnv();

const nextConfig: NextConfig = {
  experimental: {
    optimizePackageImports: ["firebase"],
    // Needed because report uploads are proxied through Next via /backend.
    middlewareClientMaxBodySize: "300mb",
    // Analyze can take several minutes for large report batches.
    proxyTimeout: 15 * 60 * 1000,
  },
  async rewrites() {
    return [
      {
        source: "/backend/:path*",
        destination: "http://127.0.0.1:8000/:path*",
      },
    ];
  },
};

export default nextConfig;