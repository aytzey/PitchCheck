import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PitchScore — Persuasion Intelligence",
  description: "Score your sales pitches with neural persuasion analysis",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
