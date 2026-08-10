"use client";

import {
  ArrowRight,
  BookOpen,
  ChevronLeft,
  ChevronRight,
  Code2,
  ExternalLink,
  Home,
  Languages,
  Menu,
  Search,
  X,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import publication from "../PUBLICATION.json";
import content from "./generated/content.json";

type Language = "ru" | "en";
type CourseKey = "modern-llms" | "rl-for-llm" | "information-theory-for-ml";

type Module = {
  id: string;
  module_key: string;
  position: number;
  title: string;
  content_file: string;
  characters: number;
  image_count: number;
};

type Course = {
  course_key: CourseKey;
  language: Language;
  title: string;
  description: string;
  stepik_url: string;
  module_count: number;
  characters: number;
  modules: Module[];
};

const courses = content.courses as Course[];

const copy = {
  ru: {
    brand: "Открытые курсы",
    author: "Влад Куликов",
    home: "Главная",
    eyebrow: "AI / ML · LLM · Образование",
    heroTitle: "Влад Куликов",
    heroText:
      "Lead AI/ML (LLMs) в FinTech. С 2017 года занимаюсь R&D, разработкой ML и GenAI-систем и консультированием бизнеса по внедрению AI.",
    viewCourses: "Перейти к курсам",
    stepikProfile: "Профиль на Stepik",
    linkedIn: "LinkedIn",
    aboutTitle: "О курсах",
    aboutText:
      "Цель этой серии — систематизировать сложные темы и помочь выстроить их понимание на разной глубине: от инженерной интуиции и реализации в коде до математических принципов каждого механизма.",
    education: "Образование и исследования",
    educationText:
      "Физтех Харьковского университета; магистратура МФТИ, ФПМИ — «Современные методы искусственного интеллекта». Диссертация и публикации — LLMs, Learning and Reasoning at the Inference Stage.",
    seriesEyebrow: "Развивающаяся серия",
    seriesTitle: "Текущие курсы",
    seriesText:
      "Сейчас в серии три самостоятельных курса. Каждый можно проходить отдельно; новые курсы и открытые материалы будут добавляться.",
    start: "Открыть курс",
    stepik: "Полный курс на Stepik",
    modules: "модулей",
    illustrations: "иллюстраций",
    library: "Библиотека",
    search: "Найти модуль",
    contents: "Содержание",
    previous: "Предыдущий",
    next: "Следующий",
    loading: "Загружаю лекцию…",
    error: "Не удалось открыть текст лекции.",
    openSource: "Исходник сайта",
    licenses: "Лицензии",
    citation: "Цитирование",
    edition: "Редакция",
    boundary:
      "В этой версии публикуются только лекционные тексты и изображения. Проверочные материалы и практические ноутбуки остаются частью полного курса.",
    allMaterials: "Все материалы",
    readTime: "около",
    hours: "ч чтения",
    noResults: "Ничего не найдено",
    menu: "Открыть содержание",
  },
  en: {
    brand: "Open courses",
    author: "Vlad Kulikov",
    home: "Home",
    eyebrow: "AI / ML · LLM · Education",
    heroTitle: "Vlad Kulikov",
    heroText:
      "Lead AI/ML (LLMs) in FinTech. Since 2017, I have worked in R&D, built ML and GenAI systems, and advised businesses on AI adoption.",
    viewCourses: "Explore the courses",
    stepikProfile: "Stepik profile",
    linkedIn: "LinkedIn",
    aboutTitle: "About the courses",
    aboutText:
      "The goal of this series is to organize difficult subjects and make them understandable at several depths—from engineering intuition and code to the mathematical principles behind each mechanism.",
    education: "Education and research",
    educationText:
      "Physics and Technology at Kharkiv University; MIPT University master's program in Modern Artificial Intelligence. Thesis and publications: LLMs, Learning and Reasoning at the Inference Stage.",
    seriesEyebrow: "An evolving series",
    seriesTitle: "Current courses",
    seriesText:
      "The series currently includes three independent courses. Each stands on its own; new courses and open materials will be added over time.",
    start: "Open course",
    stepik: "Complete course on Stepik",
    modules: "modules",
    illustrations: "illustrations",
    library: "Library",
    search: "Find a module",
    contents: "Contents",
    previous: "Previous",
    next: "Next",
    loading: "Loading lecture…",
    error: "The lecture text could not be opened.",
    openSource: "Site source",
    licenses: "Licenses",
    citation: "Citation",
    edition: "Edition",
    boundary:
      "This edition contains lecture text and images only. Assessments and practical notebooks remain part of the complete course.",
    allMaterials: "All materials",
    readTime: "about",
    hours: "h reading",
    noResults: "No modules found",
    menu: "Open contents",
  },
} as const;

const courseNumbers: Record<CourseKey, string> = {
  "modern-llms": "01",
  "rl-for-llm": "02",
  "information-theory-for-ml": "03",
};

function getBaseUrl() {
  const value = import.meta.env.BASE_URL || "/";
  return value.endsWith("/") ? value : `${value}/`;
}

function parseLocation() {
  if (typeof window === "undefined") {
    return { language: "en" as Language, courseKey: null, moduleKey: null };
  }
  const params = new URLSearchParams(window.location.search);
  const language = params.get("lang") === "ru" ? "ru" : "en";
  const courseKey = params.get("course") as CourseKey | null;
  const moduleKey = params.get("module");
  return { language, courseKey, moduleKey };
}

function courseFor(courseKey: CourseKey, language: Language) {
  return courses.find(
    (course) => course.course_key === courseKey && course.language === language,
  );
}

function formatModuleNumber(moduleKey: string) {
  return moduleKey.replace("module-", "").toUpperCase();
}

function slugify(value: string) {
  return value
    .toLowerCase()
    .replace(/[`*_()[\]{}]/g, "")
    .replace(/[^a-zа-яё0-9]+/gi, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 80);
}

function findInternalModule(href: string | undefined) {
  if (!href || /^https?:|^mailto:|^#/.test(href)) return null;
  const match = href.match(/module[-_](\d{1,2}b?)/i);
  if (!match) return null;
  return `module-${match[1].toLowerCase().padStart(2, "0")}`;
}

export function CourseLibrary() {
  const initial = parseLocation();
  const [language, setLanguage] = useState<Language>(initial.language);
  const [courseKey, setCourseKey] = useState<CourseKey | null>(initial.courseKey);
  const [moduleKey, setModuleKey] = useState<string | null>(initial.moduleKey);
  const [markdown, setMarkdown] = useState("");
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState(false);
  const [query, setQuery] = useState("");
  const [drawerOpen, setDrawerOpen] = useState(false);
  const t = copy[language];

  const activeCourse = useMemo(
    () => (courseKey ? courseFor(courseKey, language) : undefined),
    [courseKey, language],
  );
  const activeModule = useMemo(
    () => activeCourse?.modules.find((module) => module.module_key === moduleKey),
    [activeCourse, moduleKey],
  );

  useEffect(() => {
    document.documentElement.lang = language;
  }, [language]);

  const updateLocation = useCallback(
    (nextLanguage: Language, nextCourse: CourseKey | null, nextModule: string | null, push = true) => {
      if (typeof window === "undefined") return;
      const params = new URLSearchParams();
      params.set("lang", nextLanguage);
      if (nextCourse) params.set("course", nextCourse);
      if (nextModule) params.set("module", nextModule);
      const url = `${window.location.pathname}?${params.toString()}`;
      window.history[push ? "pushState" : "replaceState"]({}, "", url);
    },
    [],
  );

  const selectModule = useCallback(
    (nextCourse: CourseKey, nextModule: string, push = true) => {
      setCourseKey(nextCourse);
      setModuleKey(nextModule);
      setDrawerOpen(false);
      updateLocation(language, nextCourse, nextModule, push);
      window.scrollTo({ top: 0, behavior: "smooth" });
    },
    [language, updateLocation],
  );

  const goHome = useCallback(() => {
    setCourseKey(null);
    setModuleKey(null);
    setDrawerOpen(false);
    updateLocation(language, null, null);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, [language, updateLocation]);

  const switchLanguage = useCallback(() => {
    const nextLanguage: Language = language === "ru" ? "en" : "ru";
    setLanguage(nextLanguage);
    updateLocation(nextLanguage, courseKey, moduleKey, false);
  }, [courseKey, language, moduleKey, updateLocation]);

  useEffect(() => {
    const onPopState = () => {
      const next = parseLocation();
      setLanguage(next.language);
      setCourseKey(next.courseKey);
      setModuleKey(next.moduleKey);
    };
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  useEffect(() => {
    if (!activeModule) {
      setMarkdown("");
      return;
    }
    const controller = new AbortController();
    setLoading(true);
    setLoadError(false);
    fetch(`${getBaseUrl()}content/${activeModule.content_file}`, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.text();
      })
      .then((value) => setMarkdown(value))
      .catch((error) => {
        if (error.name !== "AbortError") setLoadError(true);
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [activeModule]);

  const visibleModules = useMemo(() => {
    if (!activeCourse) return [];
    const needle = query.trim().toLocaleLowerCase(language);
    if (!needle) return activeCourse.modules;
    return activeCourse.modules.filter(
      (module) =>
        module.title.toLocaleLowerCase(language).includes(needle) ||
        module.module_key.includes(needle),
    );
  }, [activeCourse, language, query]);

  const moduleIndex = activeCourse && activeModule ? activeCourse.modules.indexOf(activeModule) : -1;
  const previousModule = moduleIndex > 0 ? activeCourse?.modules[moduleIndex - 1] : undefined;
  const nextModule =
    activeCourse && moduleIndex >= 0 && moduleIndex < activeCourse.modules.length - 1
      ? activeCourse.modules[moduleIndex + 1]
      : undefined;
  const displayMarkdown = markdown.replace(/^#\s+.+?\r?\n+/, "");
  const headings = useMemo(
    () =>
      [...displayMarkdown.matchAll(/^##\s+(.+)$/gm)]
        .map((match) => match[1].replace(/[`*_]/g, ""))
        .slice(0, 24),
    [displayMarkdown],
  );

  const openInternalModule = useCallback(
    (target: string) => {
      if (!courseKey || !activeCourse?.modules.some((module) => module.module_key === target)) return;
      selectModule(courseKey, target);
    },
    [activeCourse, courseKey, selectModule],
  );

  return (
    <div className="site-shell" data-course={courseKey ?? "home"}>
      <header className="topbar">
        <button className="brand" onClick={goHome} aria-label={t.home}>
          <span className="brand-mark">VK</span>
          <span>
            <strong>{t.author}</strong>
            <small>{t.brand}</small>
          </span>
        </button>
        <nav className="top-actions" aria-label="Site navigation">
          {activeCourse && (
            <button className="mobile-menu" onClick={() => setDrawerOpen(true)} aria-label={t.menu}>
              <Menu size={19} />
            </button>
          )}
          <a href="https://github.com/VladGKulikov/open-courses" target="_blank" rel="noreferrer">
            <Code2 size={17} />
            <span className="desktop-label">GitHub</span>
          </a>
          <button className="language-toggle" onClick={switchLanguage} aria-label="Switch language">
            <Languages size={17} />
            {language === "ru" ? "EN" : "RU"}
          </button>
        </nav>
      </header>

      {!activeCourse || !activeModule ? (
        <main className="home-view">
          <section className="hero">
            <div className="hero-copy">
              <p className="eyebrow">{t.eyebrow}</p>
              <h1>{t.heroTitle}</h1>
              <p className="hero-lead">{t.heroText}</p>
              <div className="hero-education">
                <p className="eyebrow">{t.education}</p>
                <p>{t.educationText}</p>
              </div>
              <div className="hero-actions">
                <a className="primary-link" href="#courses">
                  {t.viewCourses}
                  <ArrowRight size={18} />
                </a>
                <a href="https://stepik.org/users/29821475" target="_blank" rel="noreferrer">
                  {t.stepikProfile}
                  <ExternalLink size={16} />
                </a>
                <a href="https://www.linkedin.com/in/vlad-g-kulikoff/" target="_blank" rel="noreferrer">
                  {t.linkedIn}
                  <ExternalLink size={16} />
                </a>
              </div>
            </div>
            <aside className="hero-about" aria-labelledby="about-title">
              <div className="about-intro">
                <p className="eyebrow">{t.aboutTitle}</p>
                <h2 id="about-title">{t.aboutText}</h2>
              </div>
            </aside>
          </section>

          <section id="courses" className="catalog" aria-labelledby="catalog-title">
            <div className="section-heading">
              <div>
                <p className="eyebrow">{t.seriesEyebrow}</p>
                <h2 id="catalog-title">{t.seriesTitle}</h2>
              </div>
              <p className="series-description">{t.seriesText}</p>
            </div>
            <div className="course-grid">
              {courses
                .filter((course) => course.language === language)
                .map((course) => {
                  const imageCount = course.modules.reduce((sum, module) => sum + module.image_count, 0);
                  const readingHours = Math.max(1, Math.round(course.characters / 60000));
                  return (
                    <article className="course-card" key={course.course_key} data-accent={course.course_key}>
                      <div className="course-card-top">
                        <span className="course-number">{courseNumbers[course.course_key]}</span>
                        <BookOpen size={22} />
                      </div>
                      <h3>{course.title}</h3>
                      <p>{course.description}</p>
                      <dl className="course-stats">
                        <div>
                          <dt>{course.module_count}</dt>
                          <dd>{t.modules}</dd>
                        </div>
                        <div>
                          <dt>{imageCount}</dt>
                          <dd>{t.illustrations}</dd>
                        </div>
                        <div>
                          <dt>{readingHours}</dt>
                          <dd>{t.hours}</dd>
                        </div>
                      </dl>
                      <button
                        className="course-open"
                        onClick={() => selectModule(course.course_key, course.modules[0].module_key)}
                      >
                        {t.start}
                        <ArrowRight size={18} />
                      </button>
                      <a
                        className="course-stepik"
                        href={course.stepik_url}
                        target="_blank"
                        rel="noreferrer"
                      >
                        {t.stepik}
                        <ExternalLink size={15} />
                      </a>
                    </article>
                  );
                })}
            </div>
          </section>

          <aside className="boundary-note">
            <span>PUBLIC / LECTURES</span>
            <p>{t.boundary}</p>
          </aside>
        </main>
      ) : (
        <main className="reader-layout">
          <aside className={`course-sidebar ${drawerOpen ? "is-open" : ""}`}>
            <div className="sidebar-head">
              <button className="back-home" onClick={goHome}>
                <Home size={16} />
                {t.home}
              </button>
              <button className="drawer-close" onClick={() => setDrawerOpen(false)} aria-label="Close">
                <X size={20} />
              </button>
              <span className="sidebar-course-number">{courseNumbers[activeCourse.course_key]}</span>
              <h2>{activeCourse.title}</h2>
              <a href={activeCourse.stepik_url} target="_blank" rel="noreferrer">
                {t.stepik}
                <ExternalLink size={14} />
              </a>
            </div>
            <label className="module-search">
              <Search size={16} />
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder={t.search}
              />
            </label>
            <nav className="module-list" aria-label={t.contents}>
              {visibleModules.length ? (
                visibleModules.map((module) => (
                  <button
                    key={module.id}
                    className={module.id === activeModule.id ? "active" : ""}
                    onClick={() => selectModule(activeCourse.course_key, module.module_key)}
                  >
                    <span>{formatModuleNumber(module.module_key)}</span>
                    <strong>{module.title.replace(/^(Module|Модуль)\s+\d+b?[.:]?\s*/i, "")}</strong>
                  </button>
                ))
              ) : (
                <p className="empty-search">{t.noResults}</p>
              )}
            </nav>
          </aside>
          {drawerOpen && <button className="drawer-scrim" onClick={() => setDrawerOpen(false)} aria-label="Close" />}

          <article className="reading-column">
            <div className="reading-meta">
              <span>{activeCourse.title}</span>
              <span>
                {moduleIndex + 1} / {activeCourse.modules.length}
              </span>
            </div>
            <h1 className="reading-title">{activeModule.title}</h1>
            <div className="reading-rule">
              <span style={{ width: `${((moduleIndex + 1) / activeCourse.modules.length) * 100}%` }} />
            </div>

            {loading && <div className="lecture-state">{t.loading}</div>}
            {loadError && <div className="lecture-state error">{t.error}</div>}
            {!loading && !loadError && (
              <div className="lecture-prose">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm, remarkMath]}
                  rehypePlugins={[rehypeKatex]}
                  components={{
                    h2: ({ children }) => {
                      const label = String(children);
                      return <h2 id={slugify(label)}>{children}</h2>;
                    },
                    h3: ({ children }) => {
                      const label = String(children);
                      return <h3 id={slugify(label)}>{children}</h3>;
                    },
                    a: ({ href, children, ...props }) => {
                      const internalModule = findInternalModule(href);
                      if (internalModule && activeCourse.modules.some((module) => module.module_key === internalModule)) {
                        return (
                          <a
                            href={`?lang=${language}&course=${activeCourse.course_key}&module=${internalModule}`}
                            onClick={(event) => {
                              event.preventDefault();
                              openInternalModule(internalModule);
                            }}
                          >
                            {children}
                          </a>
                        );
                      }
                      const external = Boolean(href && /^https?:/.test(href));
                      return (
                        <a
                          href={href}
                          target={external ? "_blank" : undefined}
                          rel={external ? "noreferrer" : undefined}
                          {...props}
                        >
                          {children}
                        </a>
                      );
                    },
                    img: ({ src, alt }) => {
                      const value = String(src ?? "");
                      const resolved = /^https?:|^data:/.test(value)
                        ? value
                        : `${getBaseUrl()}${value.replace(/^\.\//, "")}`;
                      return <img src={resolved} alt={alt ?? ""} loading="lazy" />;
                    },
                    table: ({ children }) => (
                      <div className="table-scroll">
                        <table>{children}</table>
                      </div>
                    ),
                  }}
                >
                  {displayMarkdown}
                </ReactMarkdown>
              </div>
            )}

            <nav className="module-pagination" aria-label="Module navigation">
              {previousModule ? (
                <button onClick={() => selectModule(activeCourse.course_key, previousModule.module_key)}>
                  <ChevronLeft size={18} />
                  <span>
                    <small>{t.previous}</small>
                    {formatModuleNumber(previousModule.module_key)}
                  </span>
                </button>
              ) : (
                <span />
              )}
              {nextModule && (
                <button onClick={() => selectModule(activeCourse.course_key, nextModule.module_key)}>
                  <span>
                    <small>{t.next}</small>
                    {formatModuleNumber(nextModule.module_key)}
                  </span>
                  <ChevronRight size={18} />
                </button>
              )}
            </nav>
          </article>

          <aside className="toc-rail">
            <p>{t.contents}</p>
            <nav>
              {headings.map((heading, index) => (
                <a key={`${heading}-${index}`} href={`#${slugify(heading)}`}>
                  {heading.replace(/^\d+[.:]\s*/, "")}
                </a>
              ))}
            </nav>
          </aside>
        </main>
      )}

      <footer className="site-footer">
        <span>© 2026 {t.author} · {t.edition} {publication.edition}</span>
        <nav aria-label="Publication information">
          <a
            href="https://github.com/VladGKulikov/open-courses/blob/main/LICENSE.md"
            target="_blank"
            rel="noreferrer"
          >
            {t.licenses}
            <ExternalLink size={13} />
          </a>
          <a
            href="https://github.com/VladGKulikov/open-courses/blob/main/CITATION.cff"
            target="_blank"
            rel="noreferrer"
          >
            {t.citation}
            <ExternalLink size={13} />
          </a>
          <a href="https://github.com/VladGKulikov/open-courses" target="_blank" rel="noreferrer">
            {t.openSource}
            <ExternalLink size={13} />
          </a>
        </nav>
      </footer>
    </div>
  );
}
