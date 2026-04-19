import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL("https://github.com/aytzey/PitchCheck"),
  title: "PitchCheck - Neural Persuasion Intelligence",
  description:
    "Score sales pitches with TRIBE neural signals, persona-aware LLM analysis, local GPU acceleration, and Vast.ai fallback.",
  manifest: "/site.webmanifest",
  icons: {
    icon: "/favicon.svg",
  },
  openGraph: {
    title: "PitchCheck",
    description:
      "Desktop-first neural persuasion scoring with local GPU and Vast.ai fallback.",
    images: ["/og.svg"],
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
