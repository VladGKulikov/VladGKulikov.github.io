import type { Metadata } from "next";

const siteUrl = new URL("https://vlad-kulikov-open-courses.gen-ai-vgk.chatgpt.site");

const languageAlternates = {
  en: "/",
  ru: "/ru/",
  "x-default": "/",
};

export const englishMetadata: Metadata = {
  metadataBase: siteUrl,
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
