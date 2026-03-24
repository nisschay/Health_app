import type { Metadata } from "next";
import { Sora } from "next/font/google";
import { AuthProvider } from "@/lib/auth-context";

import "./globals.css";

const sora = Sora({ subsets: ["latin"], display: "swap" });

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
    <html lang="en">
      <body className={sora.className}>
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}