import type { NextConfig } from "next";

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