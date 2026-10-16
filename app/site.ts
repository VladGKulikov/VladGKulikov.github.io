import content from "./generated/content.json";
import config from "../SITE_CONFIG.json";

export type Language = "ru" | "en";
export type CourseKey = "modern-llms" | "rl-for-llm" | "information-theory-for-ml";
export type CourseModule = typeof content.courses[number]["modules"][number];
export type Course = Omit<typeof content.courses[number], "language" | "course_key"> & { language: Language; course_key: CourseKey };
export type PageRoute = { language: Language; courseKey: CourseKey | null; moduleKey: string | null; notFound?: boolean };
export const courses = content.courses as Course[];
export const siteOrigin = config.origin;

export function buildLocation(language: Language, course: string | null = null, module: string | null = null) {
  return `${language === "ru" ? "/ru/" : "/"}${course ? `courses/${course}/${module ? `${module}/` : ""}` : ""}`;
}

export function parseUrl(url: URL): PageRoute {
  const parts = url.pathname.split("/").filter(Boolean);
  const language: Language = parts[0] === "ru" ? "ru" : "en";
  if (language === "ru") parts.shift();
  if (parts.at(-1) === "index.html") parts.pop();
  if (!parts.length) {
    const legacyLanguage = url.searchParams.get("lang");
    const selectedLanguage = legacyLanguage === "en" || legacyLanguage === "ru" ? legacyLanguage : language;
    const courseKey = url.searchParams.get("course") as CourseKey | null;
    const moduleKey = url.searchParams.get("module");
    const course = courses.find(c => c.language === selectedLanguage && c.course_key === courseKey);
    if (courseKey && !course || moduleKey && !course?.modules.some(m => m.module_key === moduleKey)) {
      return { language: selectedLanguage, courseKey: null, moduleKey: null, notFound: true };
    }
    // Earlier course-only links opened the first lecture, rather than an overview.
    return { language: selectedLanguage, courseKey, moduleKey: moduleKey ?? course?.modules[0]?.module_key ?? null };
  }
  const course = courses.find(c => c.language === language && c.course_key === parts[1]);
  if (parts[0] !== "courses" || parts.length > 3 || !course || parts[2] && !course.modules.some(m => m.module_key === parts[2])) {
    return { language, courseKey: null, moduleKey: null, notFound: true };
  }
  return { language, courseKey: course.course_key, moduleKey: parts[2] ?? null };
}

export function pageMetadata(route: PageRoute) {
  const course = courses.find(c => c.language === route.language && c.course_key === route.courseKey);
  const lecture = course?.modules.find(m => m.module_key === route.moduleKey);
  const ru = route.language === "ru";
  const title = route.notFound ? (ru ? "Страница не найдена" : "Page not found")
    : lecture ? `${lecture.title} · ${course!.title} · ${ru ? "Влад Куликов" : "Vlad Kulikov"}`
    : course ? `${course.title} — ${ru ? "курс и оглавление" : "course and contents"} · ${ru ? "Влад Куликов" : "Vlad Kulikov"}`
    : ru ? "Открытые курсы Влада Куликова" : "Vlad Kulikov Open Courses";
  const description = lecture ? (ru
    ? `${lecture.title}. ${course!.title}: открытая лекция Влада Куликова с математическими формулами и иллюстрациями.`
    : `${lecture.title}. ${course!.title}: an open lecture by Vlad Kulikov with mathematical formulas and illustrations.`)
    : course ? course.description
    : ru ? "Три углублённых AI/ML-курса на русском и английском: Modern LLMs, RL для LLM и теория информации для машинного обучения."
    : "Three in-depth AI/ML courses in English and Russian: Modern LLMs, RL for LLM, and Information Theory for Machine Learning.";
  const path = buildLocation(route.language, route.courseKey, route.moduleKey);
  return { title, description, path, canonical: `${siteOrigin}${path}`, language: route.language,
    en: `${siteOrigin}${buildLocation("en", route.courseKey, route.moduleKey)}`,
    ru: `${siteOrigin}${buildLocation("ru", route.courseKey, route.moduleKey)}`,
    image: `${siteOrigin}/og.png`, notFound: Boolean(route.notFound) };
}

export function updatePageMetadata(route: PageRoute) {
  const meta = pageMetadata(route);
  document.documentElement.lang = route.language;
  document.title = meta.title;
  const set = (selector: string, attribute: string, value: string) => document.querySelector(selector)?.setAttribute(attribute, value);
  set('meta[name="description"]', "content", meta.description);
  set('link[rel="canonical"]', "href", meta.canonical);
  for (const lang of ["en", "ru", "x-default"]) set(`link[hreflang="${lang}"]`, "href", lang === "ru" ? meta.ru : meta.en);
  for (const prefix of ["og", "twitter"]) {
    const attr = prefix === "og" ? "property" : "name";
    for (const field of ["title", "description", "image"] as const) set(`meta[${attr}="${prefix}:${field}"]`, "content", meta[field]);
  }
  set('meta[property="og:url"]', "content", meta.canonical);
  set('meta[property="og:locale"]', "content", route.language === "ru" ? "ru_RU" : "en_US");
  set('meta[property="og:locale:alternate"]', "content", route.language === "ru" ? "en_US" : "ru_RU");
}
