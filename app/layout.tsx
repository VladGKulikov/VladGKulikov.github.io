import type { Metadata } from "next";
import "katex/dist/katex.min.css";
import "./globals.css";

export const metadata: Metadata = {
  title: "Vlad Kulikov · Open Courses",
  description:
    "Личная страница Влада Куликова и открытая библиотека курсов по современным LLM, reinforcement learning и теории информации для машинного обучения.",
  openGraph: {
    title: "Vlad Kulikov · Open Courses",
    description:
      "Три двуязычных курса: Modern LLMs, RL for LLM и Information Theory for ML.",
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
    <html lang="ru">
      <body>{children}</body>
    </html>
  );
}
