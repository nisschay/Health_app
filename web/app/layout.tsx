import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "Medical Project",
  description: "AI-powered medical report analysis with a Vercel-ready frontend and Python backend."
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}