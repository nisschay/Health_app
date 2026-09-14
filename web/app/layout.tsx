import type { Metadata } from "next";
import { DM_Sans, JetBrains_Mono, Playfair_Display } from "next/font/google";
import { AuthProvider } from "@/lib/auth-context";
import BackendWakeupOverlay from "./BackendWakeupOverlay";

import "./globals.css";

const bodyFont = DM_Sans({ subsets: ["latin"], weight: ["300", "400", "500", "600"], variable: "--font-body", display: "swap" });
const displayFont = Playfair_Display({ subsets: ["latin"], weight: ["600", "700"], variable: "--font-display", display: "swap" });
const monoFont = JetBrains_Mono({ subsets: ["latin"], weight: ["400", "500"], variable: "--font-mono", display: "swap" });

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
