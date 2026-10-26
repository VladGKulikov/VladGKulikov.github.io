import type { Metadata } from "next";

const siteUrl = new URL("https://vladkulikov.com");

const icons: Metadata["icons"] = {
  icon: [
    { url: "/favicon.ico?v=20261026", sizes: "16x16 32x32 48x48" },
    { url: "/favicon.svg?v=20261026", type: "image/svg+xml", sizes: "any" },
  ],
  apple: [{ url: "/apple-touch-icon.png?v=20261026", sizes: "180x180" }],
};

const languageAlternates = {
  en: "/",
  ru: "/ru/",
  "x-default": "/",
};

export const englishMetadata: Metadata = {
  metadataBase: siteUrl,
  icons,
  title: "Vlad Kulikov Open Courses",
  description:
    "Three in-depth AI/ML courses in English and Russian: Modern LLMs, RL for LLM, and Information Theory for Machine Learning.",
  alternates: {
    canonical: "/",
    languages: languageAlternates,
  },
  openGraph: {
    title: "Vlad Kulikov Open Courses",
    description:
      "Three in-depth AI/ML courses in English and Russian: Modern LLMs, RL for LLM, and Information Theory for Machine Learning.",
    type: "website",
    url: "/",
    locale: "en_US",
    alternateLocale: ["ru_RU"],
    images: [{ url: "/og.png", width: 1732, height: 910 }],
  },
  twitter: {
    card: "summary_large_image",
    title: "Vlad Kulikov Open Courses",
    description:
      "Three in-depth AI/ML courses in English and Russian: Modern LLMs, RL for LLM, and Information Theory for Machine Learning.",
    images: ["/og.png"],
  },
};

export const russianMetadata: Metadata = {
  metadataBase: siteUrl,
  icons,
  title: "Открытые курсы Влада Куликова",
  description:
    "Три углублённых AI/ML-курса на русском и английском: Modern LLMs, RL для LLM и теория информации для машинного обучения.",
  alternates: {
    canonical: "/ru/",
    languages: languageAlternates,
  },
  openGraph: {
    title: "Открытые курсы Влада Куликова",
    description:
      "Три углублённых AI/ML-курса на русском и английском: Modern LLMs, RL для LLM и теория информации для машинного обучения.",
    type: "website",
    url: "/ru/",
    locale: "ru_RU",
    alternateLocale: ["en_US"],
    images: [{ url: "/og.png", width: 1732, height: 910 }],
  },
  twitter: {
    card: "summary_large_image",
    title: "Открытые курсы Влада Куликова",
    description:
      "Три углублённых AI/ML-курса на русском и английском: Modern LLMs, RL для LLM и теория информации для машинного обучения.",
    images: ["/og.png"],
  },
};
