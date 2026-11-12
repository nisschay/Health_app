import type { Metadata } from "next";
import localFont from "next/font/local";
import { AuthProvider } from "@/lib/auth-context";
import BackendWakeupOverlay from "./BackendWakeupOverlay";

import "./globals.css";

// Fonts ship with the app. next/font/google downloaded them during the build, and when
// Google answered badly the Vercel build failed ("An error occurred in next/font").
const bodyFont = localFont({ src: "../node_modules/@fontsource-variable/dm-sans/files/dm-sans-latin-wght-normal.woff2", weight: "300 600", variable: "--font-body", display: "swap" });
const displayFont = localFont({ src: "../node_modules/@fontsource-variable/playfair-display/files/playfair-display-latin-wght-normal.woff2", weight: "600 700", variable: "--font-display", display: "swap" });
const monoFont = localFont({ src: "../node_modules/@fontsource-variable/jetbrains-mono/files/jetbrains-mono-latin-wght-normal.woff2", weight: "400 500", variable: "--font-mono", display: "swap" });

export const metadata: Metadata = {
  title: "Medical Report Analyzer",
  description: "AI-powered medical report analysis – upload, analyze, and track your health over time."
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${bodyFont.variable} ${displayFont.variable} ${monoFont.variable}`}>
      <body>
        <AuthProvider>{children}</AuthProvider>
        <BackendWakeupOverlay />
      </body>
    </html>
  );
}
