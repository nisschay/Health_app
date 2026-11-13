import type { Metadata } from "next";
import localFont from "next/font/local";
import { AuthProvider } from "@/lib/auth-context";
import BackendWakeupOverlay from "./BackendWakeupOverlay";

import "./globals.css";

// Fonts ship with the app. next/font/google downloaded them during the build, and when
// Google answered badly the Vercel build failed ("An error occurred in next/font").
const bodyFont = localFont({
  src: "../node_modules/@fontsource-variable/figtree/files/figtree-latin-wght-normal.woff2",
  weight: "300 900",
  variable: "--font-body-face",
  display: "swap",
});
const displayFont = localFont({
  src: "../node_modules/@fontsource-variable/newsreader/files/newsreader-latin-wght-normal.woff2",
  weight: "200 800",
  variable: "--font-display-face",
  display: "swap",
});

export const metadata: Metadata = {
  title: "LabTrail",
  description:
    "Follow your family’s lab results over time. Not medical advice.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en-IN"
      suppressHydrationWarning
      className={`${bodyFont.variable} ${displayFont.variable}`}
    >
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `try { const t = localStorage.getItem("health-app.theme"); if(t === "light" || t === "dark") document.documentElement.dataset.theme = t; } catch {}`,
          }}
        />
      </head>
      <body>
        <AuthProvider>{children}</AuthProvider>
        <BackendWakeupOverlay />
      </body>
    </html>
  );
}
