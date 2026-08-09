import type { Metadata } from "next";
import "katex/dist/katex.min.css";
import "./globals.css";

export const metadata: Metadata = {
  title: "Vlad Kulikov · Open Courses",
  description:
    "Vlad Kulikov's personal website and an open bilingual course library covering modern LLMs, reinforcement learning, and information theory for machine learning.",
  openGraph: {
    title: "Vlad Kulikov · Open Courses",
    description:
      "An evolving bilingual course series: Modern LLMs, RL for LLM, Information Theory for ML, and more to come.",
    type: "website",
    images: [{ url: "/og.png", width: 1732, height: 910 }],
  },
  twitter: {
    card: "summary_large_image",
    title: "Vlad Kulikov · Open Courses",
    description: "Modern LLMs · RL for LLM · Information Theory for ML",
    images: ["/og.png"],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
