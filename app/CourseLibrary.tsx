"use client";
/* eslint-disable @next/next/no-html-link-for-pages, @next/next/no-img-element -- Static hosting uses native links and image files. */

import {
  ArrowRight,
  BookOpen,
  ChevronLeft,
  ChevronRight,
  Code2,
  ExternalLink,
  Home,
  Languages,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
  Search,
  X,
} from "lucide-react";
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import publication from "../PUBLICATION.json";
import { markdownHeadings } from "./markdown-headings";
import { module20Equation } from "./module-20-equations";
import { captureReadingPosition, readingPositionAtAnchor, restoreReadingPosition, type ReadingPosition } from "./reading-position";

import config from "../SITE_CONFIG.json";
import { buildLocation, courses, parseUrl, updatePageMetadata, type Language, type CourseKey, type CourseModule as Module, type PageRoute, type PageSection } from "./site";
import { trackPage } from "./analytics";
import { quizFor, quizModules, type Quiz } from './quizzes';
import { QuizReader, QuizMarkdown, questionAnchor, questionLabel } from './QuizReader';
import { practiceFor, practiceModules, practiceAnchor, type Practice } from './practice';
import { PracticeReader, PracticeMarkdown } from './PracticeReader';

const copy = {
  ru: {
    brand: "Открытые лекции",
    author: "Влад Куликов",
    home: "Главная",
    eyebrow: "AI / ML · LLM · ОБРАЗОВАНИЕ · ТРИ УГЛУБЛЁННЫХ КУРСА · RU / EN",
    heroTitle: "ВЛАД КУЛИКОВ",
    heroText:
      "AI/ML-инженер и исследователь, AI Function Lead в FinTech. С 2017 года занимаюсь R&D, создаю ML- и GenAI-системы и консультирую компании по внедрению AI.",
    viewCourses: "Перейти к курсам",
    stepikProfile: "Профиль на Stepik",
    linkedIn: "LinkedIn",
    aboutTitle: "О курсах",
    aboutText:
      "Я создаю эти курсы в том виде, которого мне самому не хватало при изучении этих тем: как цельные маршруты от интуиции и математики к реализации и инженерной практике.",
    aboutAudience:
      "Для ML-практиков и подготовленных слушателей. Предполагаются базовая вузовская математика и Python.",
    education: "Образование и исследования",
    educationText:
      "Физика — Харьковский университет. Прикладная математика и информатика, специализация «Современные методы искусственного интеллекта» — магистратура МФТИ. Моя диссертация и публикации посвящены LLM — прежде всего обучению и рассуждению моделей на этапе инференса.",
    seriesEyebrow: "Развивающаяся серия",
    seriesTitle: "Текущие курсы",
    seriesText:
      "Сейчас в серии три самостоятельных курса. Каждый можно проходить отдельно; новые курсы и открытые материалы будут добавляться.",
    start: "Читать на сайте",
    stepik: "Открыть в Stepik",
    leanpub: "PDF / EPUB бесплатно · Поддержать автора",
    courseContentTitle: "Полный курс · Бесплатно",
    courseContentText: "Лекции, тесты с проверкой, упражнения с решениями и файлы для практики",
    quizQuestions: "Вопросы в тестах",
    exerciseItems: "Упражнения и проекты",
    notebookFiles: "Ноутбуки",
    pythonFiles: "Python-примеры и файлы решений",
    studyHere: "Учиться на сайте",
    siteHint: "Основная площадка",
    stepikHint: "Если вам удобнее Stepik",
    bookTitle: "PDF / EPUB",
    bookText: "Скачать бесплатно или поддержать автора на Leanpub",
    modules: "модулей",
    pages: "страниц в PDF",
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
    boundaryLabel: "Где учиться",
    boundary:
      "Этот сайт — основная площадка курсов. Те же материалы бесплатно доступны на Stepik: выбирайте, где вам удобнее учиться. Ноутбуки и Python-файлы на сайте можно читать и скачивать.",
    allMaterials: "Все материалы",
    noResults: "Ничего не найдено",
    modulesMenu: "Список модулей",
    outlineMenu: "Оглавление лекции",
    showModules: "Показать список модулей",
    hideModules: "Скрыть список модулей",
    showOutline: "Показать оглавление",
    hideOutline: "Скрыть оглавление",
  },
  en: {
    brand: "Public / lectures",
    author: "Vlad Kulikov",
    home: "Home",
    eyebrow: "AI / ML · LLM · EDUCATION · THREE IN-DEPTH COURSES · EN / RU",
    heroTitle: "VLAD KULIKOV",
    heroText:
      "AI/ML Engineer & Researcher — AI Function Lead in FinTech. Since 2017, I have worked in R&D, built ML and GenAI systems, and advised businesses on AI adoption.",
    viewCourses: "Explore the courses",
    stepikProfile: "Stepik profile",
    linkedIn: "LinkedIn",
    aboutTitle: "About the courses",
    aboutText:
      "I create these courses in the form I wish I had when learning the subjects myself: coherent paths from intuition and mathematics to implementation and engineering practice.",
    aboutAudience:
      "For ML practitioners and advanced learners. Basic university mathematics and Python are assumed.",
    education: "Education and research:",
    educationText:
      "Physics — Kharkiv University; M.Sc. in Applied Mathematics & Computer Science — MIPT University, “Modern State of Artificial Intelligence” program. My thesis and publications focus on LLMs — particularly on learning and reasoning at inference time.",
    seriesEyebrow: "An evolving series",
    seriesTitle: "Current courses",
    seriesText:
      "The series currently includes three independent courses. Each stands on its own; new courses and open materials will be added over time.",
    start: "Read online",
    stepik: "Open in Stepik",
    leanpub: "Free PDF / EPUB · Support the author",
    courseContentTitle: "Complete course · Free",
    courseContentText: "Lectures, checked quizzes, worked exercises and practice files",
    quizQuestions: "Quiz questions",
    exerciseItems: "Exercises and projects",
    notebookFiles: "Notebooks",
    pythonFiles: "Python examples and solution files",
    studyHere: "Study on this site",
    siteHint: "Main learning platform",
    stepikHint: "If you prefer Stepik",
    bookTitle: "PDF / EPUB",
    bookText: "Download free or support the author on Leanpub",
    modules: "modules",
    pages: "pages in PDF",
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
    boundaryLabel: "Where to study",
    boundary:
      "This site is the main home for the courses. The same materials are available free on Stepik: choose whichever platform suits you. Notebooks and Python files can be read and downloaded here.",
    allMaterials: "All materials",
    noResults: "No modules found",
    modulesMenu: "Course modules",
    outlineMenu: "Lecture outline",
    showModules: "Show course modules",
    hideModules: "Hide course modules",
    showOutline: "Show lecture outline",
    hideOutline: "Hide lecture outline",
  },
} as const;

const courseNumbers: Record<CourseKey, string> = {
  "modern-llms": "01",
  "rl-for-llm": "02",
  "information-theory-for-ml": "03",
};

const leanpubUrls: Record<CourseKey, Record<Language, string>> = {
  "modern-llms": {
    en: "https://leanpub.com/modern-llms",
    ru: "https://leanpub.com/modern-llms-ru",
  },
  "rl-for-llm": {
    en: "https://leanpub.com/reinforcement-learning-for-llm",
    ru: "https://leanpub.com/reinforcement-learning-for-llm-ru",
  },
  "information-theory-for-ml": {
    en: "https://leanpub.com/information-theory-for-ml",
    ru: "https://leanpub.com/information-theory-for-ml-ru",
  },
};

const coursePageCounts: Record<CourseKey, Record<Language, number>> = {
  "modern-llms": {
    ru: 504,
    en: 460,
  },
  "rl-for-llm": {
    ru: 286,
    en: 262,
  },
  "information-theory-for-ml": {
    ru: 335,
    en: 320,
  },
};

function getBaseUrl() { return "/"; }

function parseLocation(fallbackLanguage: Language): PageRoute {
  return typeof window === "undefined"
    ? { language: fallbackLanguage, courseKey: null, moduleKey: null }
    : parseUrl(new URL(window.location.href));
}

function courseFor(courseKey: CourseKey, language: Language) {
  return courses.find(
    (course) => course.course_key === courseKey && course.language === language,
  );
}

function formatModuleNumber(module: Module) {
  return String(module.position).padStart(2, "0");
}

function findInternalModule(href: string | undefined) {
  if (!href || /^https?:|^mailto:|^#/.test(href)) return null;
  const match = href.match(/module[-_](\d{1,2}b?)/i);
  if (!match) return null;
  return `module-${match[1].toLowerCase().padStart(2, "0")}`;
}

const subscribeNotice = () => () => {};
function readNoticePreference() {
  try { return sessionStorage.getItem("stepik-notice-dismissed") === "1"; } catch { return false; }
}
function StepikNotice({ language }: { language: Language }) {
  const [dismissed, setDismissed] = useState(false);
  const previouslyDismissed = useSyncExternalStore(subscribeNotice, readNoticePreference, () => false);
  if (!config.stepikNotice || dismissed || previouslyDismissed) return null;
  return <aside className="stepik-notice">
    <span>{language === "ru" ? "Все материалы курса доступны на этом сайте: лекции, тесты, упражнения с решениями и файлы для практики. Если вам удобнее Stepik, курс доступен и там." : "All course materials are available on this site: lectures, quizzes, worked exercises and practice files. The course is also available on Stepik if you prefer."}</span>
    <button className="js-only" aria-label={language === "ru" ? "Скрыть уведомление" : "Dismiss notice"} onClick={() => {
      setDismissed(true);
      try { sessionStorage.setItem("stepik-notice-dismissed", "1"); } catch { /* Storage is optional. */ }
    }}><X size={16} /></button>
  </aside>;
}

export function CourseLibrary({ initialLanguage = "en", initialPage, initialMarkdown = "" }: {
  initialLanguage?: Language; initialPage?: PageRoute; initialMarkdown?: string;
}) {
  const initial = initialPage ?? parseLocation(initialLanguage);
  const [language, setLanguage] = useState<Language>(initial.language);
  const [courseKey, setCourseKey] = useState<CourseKey | null>(initial.courseKey);
  const [moduleKey, setModuleKey] = useState<string | null>(initial.moduleKey);
  const [section, setSection] = useState<PageSection | undefined>(initial.section);
  const [notFound, setNotFound] = useState(Boolean(initial.notFound));
  const [markdown, setMarkdown] = useState(initialMarkdown);
  const markdownCache = useRef(new Map<string, string>(initial.moduleKey ? [[`${initial.courseKey}:${initial.language}:${initial.moduleKey}:${initial.section ?? 'lecture'}`, initialMarkdown]] : []));
  const [loading, setLoading] = useState(Boolean(initial.moduleKey && !initialMarkdown));
  const [loadError, setLoadError] = useState(false);
  const [query, setQuery] = useState("");
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [sidebarHidden, setSidebarHidden] = useState(false);
  const [tocHidden, setTocHidden] = useState(false);
  const [tocDrawerOpen, setTocDrawerOpen] = useState(false);
  const [anchorKey, setAnchorKey] = useState("");
  const pendingReadingPosition = useRef<ReadingPosition | null>(null);
  const pendingScroll = useRef<number | null>(null);
  const currentHistoryKey = useRef("");
  const currentHistoryUrl = useRef("");
  const historyPositions = useRef(new Map<string, { readingPosition: ReadingPosition | null; scrollY: number }>());
  const t = copy[language];

  const activeCourse = useMemo(
    () => (courseKey ? courseFor(courseKey, language) : undefined),
    [courseKey, language],
  );
  const activeModule = useMemo(
    () => activeCourse?.modules.find((module) => module.module_key === moduleKey),
    [activeCourse, moduleKey],
  );
  const activeQuiz = quizFor(courseKey, moduleKey, language);
  const quiz = useMemo(() => section === 'tests' && markdown ? JSON.parse(markdown) as Quiz : null, [section, markdown]);
  const activePractice = section === 'exercises' || section === 'notebooks' ? practiceFor(courseKey, moduleKey, language, section) : undefined;
  const practice = useMemo(() => (section === 'exercises' || section === 'notebooks') && markdown ? JSON.parse(markdown) as Practice : null, [section, markdown]);
  const moduleSection = (key: string) => section === 'tests' ? (quizFor(courseKey, key, language) ? section : undefined) : (section && practiceFor(courseKey, key, language, section) ? section : undefined);

  useEffect(() => {
    const previous = window.history.scrollRestoration;
    window.history.scrollRestoration = "manual";
    const readerKey = window.history.state?.readerKey ?? window.crypto.randomUUID();
    window.history.replaceState({ ...window.history.state, readerKey }, "");
    currentHistoryKey.current = readerKey;
    currentHistoryUrl.current = window.location.href;
    const onHashChange = () => {
      setAnchorKey(window.location.hash);
      if (currentHistoryUrl.current === window.location.href) return;
      const readerKey = window.crypto.randomUUID();
      window.history.replaceState({ readerKey }, "");
      currentHistoryKey.current = readerKey;
      currentHistoryUrl.current = window.location.href;
    };
    window.addEventListener("hashchange", onHashChange);
    return () => {
      window.history.scrollRestoration = previous;
      window.removeEventListener("hashchange", onHashChange);
    };
  }, []);

  useEffect(() => {
    updatePageMetadata({ language, courseKey, moduleKey, section, notFound });
    trackPage({ language, courseKey, moduleKey, section, notFound });
  }, [language, courseKey, moduleKey, section, notFound]);

  useEffect(() => {
    if (!drawerOpen && !tocDrawerOpen) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setDrawerOpen(false);
        setTocDrawerOpen(false);
      }
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [drawerOpen, tocDrawerOpen]);

  useEffect(() => {
    if (!drawerOpen && !tocDrawerOpen) return;
    const modulesScreen = window.matchMedia("(max-width: 900px)");
    const outlineScreen = window.matchMedia("(max-width: 1180px)");
    const previousOverflow = document.body.style.overflow;
    const previousPaddingRight = document.body.style.paddingRight;
    const update = () => {
      const overlayOpen = (drawerOpen && modulesScreen.matches) || (tocDrawerOpen && outlineScreen.matches);
      document.body.style.overflow = previousOverflow;
      document.body.style.paddingRight = previousPaddingRight;
      if (overlayOpen) {
        // Keep the reading column unchanged when a classic scrollbar disappears.
        // Mobile browsers with overlay scrollbars need no compensation.
        const viewportWidth = document.documentElement.getBoundingClientRect().width;
        const paddingRight = parseFloat(window.getComputedStyle(document.body).paddingRight) || 0;
        document.body.style.overflow = "hidden";
        const scrollbarWidth = document.documentElement.getBoundingClientRect().width - viewportWidth;
        if (scrollbarWidth > 0) document.body.style.paddingRight = `${paddingRight + scrollbarWidth}px`;
      }
    };
    update();
    modulesScreen.addEventListener("change", update);
    outlineScreen.addEventListener("change", update);
    return () => {
      document.body.style.overflow = previousOverflow;
      document.body.style.paddingRight = previousPaddingRight;
      modulesScreen.removeEventListener("change", update);
      outlineScreen.removeEventListener("change", update);
    };
  }, [drawerOpen, tocDrawerOpen]);

  const updateLocation = useCallback(
    (nextLanguage: Language, nextCourse: CourseKey | null, nextModule: string | null, push = true, nextSection?: PageSection) => {
      if (typeof window === "undefined") return;
      const url = buildLocation(nextLanguage, nextCourse, nextModule, nextSection);
      const readerKey = window.crypto.randomUUID();
      window.history[push ? "pushState" : "replaceState"]({ readerKey }, "", url);
      currentHistoryKey.current = readerKey;
      currentHistoryUrl.current = window.location.href;
    },
    [],
  );

  const preparePage = useCallback((next: PageRoute) => {
    const cached = markdownCache.current.get(`${next.courseKey}:${next.language}:${next.moduleKey}:${next.section ?? 'lecture'}`);
    setLoadError(false);
    if (!next.moduleKey) {
      setMarkdown("");
      setLoading(false);
    } else if (cached) {
      setMarkdown(cached);
      setLoading(false);
    } else {
      setLoading(true);
    }
  }, []);

  const navigate = useCallback((next: PageRoute, push = true, hash = "") => {
    const translated = next.language !== language && next.courseKey === courseKey && next.moduleKey === moduleKey && next.section === section;
    const bookmark = captureReadingPosition();
    historyPositions.current.set(currentHistoryKey.current, { readingPosition: bookmark, scrollY: window.scrollY });
    if (push) window.history.replaceState({ ...window.history.state, readingPosition: bookmark, scrollY: window.scrollY }, "");
    pendingReadingPosition.current = translated ? bookmark : null;
    pendingScroll.current = translated ? null : 0;
    if (!translated) {
      if (next.courseKey !== courseKey || next.moduleKey !== moduleKey || next.section !== section) setMarkdown("");
      setDrawerOpen(false);
      setTocDrawerOpen(false);
    }
    setLanguage(next.language);
    setCourseKey(next.courseKey);
    setModuleKey(next.moduleKey);
    setSection(next.section);
    setNotFound(Boolean(next.notFound));
    preparePage(next);
    updateLocation(next.language, next.courseKey, next.moduleKey, push, next.section);
    if (hash) window.history.replaceState(window.history.state, "", `${buildLocation(next.language, next.courseKey, next.moduleKey, next.section)}${hash}`);
    currentHistoryUrl.current = window.location.href;
    if (!translated) window.scrollTo({ top: 0, behavior: "instant" });
  }, [language, courseKey, moduleKey, section, updateLocation, preparePage]);

  useEffect(() => {
    const onPopState = () => {
      const next = parseLocation(initialLanguage);
      // Capture the outgoing entry before replacing its DOM, including entries
      // left using the browser's Back/Forward controls rather than a site link.
      historyPositions.current.set(currentHistoryKey.current, { readingPosition: captureReadingPosition(), scrollY: window.scrollY });
      const readerKey = window.history.state?.readerKey ?? window.crypto.randomUUID();
      const saved = historyPositions.current.get(readerKey) ?? window.history.state;
      currentHistoryKey.current = readerKey;
      currentHistoryUrl.current = window.location.href;
      window.history.replaceState({ ...window.history.state, readerKey }, "");
      pendingReadingPosition.current = saved?.readingPosition ?? null;
      pendingScroll.current = pendingReadingPosition.current || window.location.hash ? null : saved?.scrollY ?? 0;
      if (next.courseKey !== courseKey || next.moduleKey !== moduleKey || next.section !== section) setMarkdown("");
      setLanguage(next.language);
      setCourseKey(next.courseKey);
      setModuleKey(next.moduleKey);
      setSection(next.section);
      setNotFound(Boolean(next.notFound));
      preparePage(next);
    };
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, [initialLanguage, language, courseKey, moduleKey, section, preparePage]);

  useLayoutEffect(() => {
    if (pendingScroll.current === null || pendingReadingPosition.current || loading || activeModule && !markdown) return;
    window.scrollTo({ top: pendingScroll.current, behavior: "instant" });
    pendingScroll.current = null;
  }, [language, courseKey, moduleKey, markdown, loading, activeModule]);

  useEffect(() => {
    if (!activeModule) return;
    const cacheKey = `${courseKey}:${language}:${moduleKey}:${section ?? 'lecture'}`;
    const cached = markdownCache.current.get(cacheKey);
    if (cached) return;
    const controller = new AbortController();
    const file = activePractice ? `practice/${activePractice.file}?v=${activePractice.hash}` : section === 'tests' && activeQuiz ? `quizzes/${activeQuiz.file}?v=${activeQuiz.hash}` : `content/${activeModule.content_file}?v=${activeModule.content_hash}`;
    fetch(`${getBaseUrl()}${file}`, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.text();
      })
      .then((value) => {
        if (!controller.signal.aborted) {
          markdownCache.current.set(cacheKey, value);
          setMarkdown(value);
        }
      })
      .catch((error) => {
        if (!controller.signal.aborted && error.name !== "AbortError") setLoadError(true);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [activeModule, activeQuiz, activePractice, courseKey, language, moduleKey, section]);

  useLayoutEffect(() => {
    const prose = document.querySelector(".lecture-prose");
    if (loading || loadError || !markdown || !prose) return;
    let anchor: HTMLElement | null = null;
    if (!pendingReadingPosition.current && window.location.hash) {
      try { anchor = document.getElementById(decodeURIComponent(window.location.hash.slice(1))); } catch { /* Malformed old anchor. */ }
    }
    const position = pendingReadingPosition.current ?? (anchor ? readingPositionAtAnchor(anchor) : null);
    if (!position) return;
    pendingReadingPosition.current = position;
    let stopped = false;
    let frame: number | undefined;
    const restore = () => {
      if (!stopped && pendingReadingPosition.current === position) restoreReadingPosition(position);
    };
    const observer = new ResizeObserver(restore);
    const stop = () => {
      if (stopped) return;
      stopped = true;
      if (pendingReadingPosition.current === position) pendingReadingPosition.current = null;
      observer.disconnect();
      window.clearTimeout(timeout);
      if (frame !== undefined) window.cancelAnimationFrame(frame);
      for (const event of ["wheel", "touchstart", "pointerdown", "keydown"]) window.removeEventListener(event, stop);
    };
    restore();
    observer.observe(prose);
    // Load pictures above the reading position before the final adjustment.
    // Stop correcting as soon as the reader interacts with the page.
    const pictures = Array.from(prose.querySelectorAll("img"))
      .filter((picture) => picture.getBoundingClientRect().top < window.innerHeight);
    for (const picture of pictures) picture.loading = "eager";
    for (const event of ["wheel", "touchstart", "pointerdown", "keydown"]) window.addEventListener(event, stop, { passive: true });
    const timeout = window.setTimeout(stop, 5000);
    void Promise.allSettled([...pictures.map((picture) => picture.decode()), document.fonts.ready]).then(() => {
      if (stopped) return;
      frame = window.requestAnimationFrame(() => {
        restore();
        stop();
      });
    });
    return stop;
  }, [markdown, loading, loadError, anchorKey]);

  useEffect(() => {
    if (loading || !markdown || pendingReadingPosition.current || !window.location.hash) return;
    try { document.getElementById(decodeURIComponent(window.location.hash.slice(1)))?.scrollIntoView(); } catch { /* Malformed old anchor. */ }
  }, [loading, markdown]);

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
  const displayMarkdown = section ? '' : markdown.replace(/^#\s+.+?\r?\n+/, "");
  const headingIndex = useMemo(
    () => practice ? practice.items.map((item, index) => ({line: index + 1, depth: 2, label: item.title, id: practiceAnchor(practice.section, item.id)})) : quiz ? quiz.questions.map((q, index) => ({ line: index + 1, depth: 2, label: questionLabel(q, language === 'ru'), id: questionAnchor(q) })) : markdownHeadings(displayMarkdown),
    [displayMarkdown, quiz, practice, language],
  );
  const semanticHeadings = useMemo(() => {
    let previous = 1;
    const result = new Map<number, number>();
    for (const heading of headingIndex) {
      const depth = Math.min(heading.depth, previous + 1);
      result.set(heading.line, depth);
      previous = depth;
    }
    return result;
  }, [headingIndex]);
  const headingIds = useMemo(() => new Map(headingIndex.map((heading) => [heading.line, heading.id])), [headingIndex]);
  const headings = headingIndex.filter((heading) => heading.depth === 2);

  useEffect(() => {
    const followLink = (event: MouseEvent) => {
    if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const anchor = (event.target as Element).closest("a[href]") as HTMLAnchorElement | null;
    if (!anchor || anchor.target || anchor.hasAttribute("download")) return;
    const url = new URL(anchor.href);
    if (url.origin !== window.location.origin) return;
    if (url.pathname === window.location.pathname && url.hash && !url.search) {
      const saved = { readingPosition: captureReadingPosition(), scrollY: window.scrollY };
      historyPositions.current.set(currentHistoryKey.current, saved);
      window.history.replaceState({ ...window.history.state, ...saved }, "");
      return;
    }
    const next = parseUrl(url);
    if (next.notFound) return;
    event.preventDefault();
    if (loading && anchor.classList.contains("language-toggle")) return;
    navigate(next, true, url.hash);
    };
    document.addEventListener("click", followLink);
    return () => document.removeEventListener("click", followLink);
  }, [navigate, loading]);

  return (
    <div className="site-shell" data-course={courseKey ?? "home"}>
      <header className="topbar">
        <a className="brand" href={buildLocation(language)} aria-label={t.home}>
          <span className="brand-mark">VK</span>
          <span>
            <strong>{t.author}</strong>
            <small>{t.brand}</small>
          </span>
        </a>
        <nav className="top-actions" aria-label="Site navigation">
          {activeCourse && activeModule && (
            <>
              <button
                className="desktop-left-toggle"
                onClick={() => setSidebarHidden((hidden) => !hidden)}
                aria-label={sidebarHidden ? t.showModules : t.hideModules}
                title={sidebarHidden ? t.showModules : t.hideModules}
                aria-controls="course-menu"
                aria-expanded={!sidebarHidden}
              >
                {sidebarHidden ? <PanelLeftOpen size={19} /> : <PanelLeftClose size={19} />}
              </button>
              <button
                className="mobile-menu"
                onClick={() => {
                  setDrawerOpen((open) => !open);
                  setTocDrawerOpen(false);
                }}
                aria-label={drawerOpen ? t.hideModules : t.showModules}
                title={drawerOpen ? t.hideModules : t.showModules}
                aria-controls="course-menu"
                aria-expanded={drawerOpen}
              >
                {drawerOpen ? <PanelLeftClose size={19} /> : <PanelLeftOpen size={19} />}
              </button>
              <button
                className="desktop-right-toggle"
                onClick={() => setTocHidden((hidden) => !hidden)}
                aria-label={tocHidden ? t.showOutline : t.hideOutline}
                title={tocHidden ? t.showOutline : t.hideOutline}
                aria-controls="lecture-outline"
                aria-expanded={!tocHidden}
              >
                {tocHidden ? <PanelRightOpen size={19} /> : <PanelRightClose size={19} />}
              </button>
              <button
                className="mobile-right-toggle"
                onClick={() => {
                  setTocDrawerOpen((open) => !open);
                  setDrawerOpen(false);
                }}
                aria-label={tocDrawerOpen ? t.hideOutline : t.showOutline}
                title={tocDrawerOpen ? t.hideOutline : t.showOutline}
                aria-controls="lecture-outline"
                aria-expanded={tocDrawerOpen}
              >
                {tocDrawerOpen ? <PanelRightClose size={19} /> : <PanelRightOpen size={19} />}
              </button>
            </>
          )}
          <a href="https://github.com/VladGKulikov/VladGKulikov.github.io" target="_blank" rel="noreferrer">
            <Code2 size={17} />
            <span className="desktop-label">GitHub</span>
          </a>
          <a className="language-toggle" href={buildLocation(language === "ru" ? "en" : "ru", courseKey, moduleKey, section)} hrefLang={language === "ru" ? "en" : "ru"} aria-label="Switch language" aria-disabled={loading}>
            <Languages size={17} />
            {language === "ru" ? "EN" : "RU"}
          </a>
        </nav>
      </header>

      {activeCourse && activeModule && <noscript>
        <style>{`.desktop-left-toggle,.desktop-right-toggle,.mobile-menu,.mobile-right-toggle,.module-search,.drawer-close,.js-only{display:none!important}`}</style>
        <nav className="static-navigation" aria-label={t.contents}>
          <details><summary>{t.modulesMenu}</summary>{activeCourse.modules.map(module => <a key={module.id} href={buildLocation(language, courseKey, module.module_key, moduleSection(module.module_key))}>{formatModuleNumber(module)} {module.title}</a>)}</details>
          <details><summary>{t.outlineMenu}</summary>{headings.map(heading => <a key={heading.id} href={`#${heading.id}`}>{heading.label}</a>)}</details>
        </nav>
      </noscript>}

      {notFound ? (
        <main className="course-overview reading-column"><h1 className="reading-title">404 · {language === "ru" ? "Страница не найдена" : "Page not found"}</h1>
          <p>{language === "ru" ? "Проверьте адрес или перейдите к библиотеке курсов." : "Check the address or return to the course library."}</p>
          <a href="/">English library</a> · <a href="/ru/">Библиотека на русском</a>
        </main>
      ) : activeCourse && !activeModule ? (
        <main className="course-overview reading-column">
          <p className="eyebrow"><a href={buildLocation(language)}>{t.library}</a></p>
          <h1 className="reading-title">{activeCourse.title}</h1>
          <div className="lecture-prose"><p>{activeCourse.description}</p></div>
          <StepikNotice language={language} />
          <p><a className="primary-link" href={buildLocation(language, courseKey, activeCourse.modules[0].module_key)}>{t.start} <ArrowRight size={18} /></a></p>
          <h2>{t.contents}</h2>
          <ol className="course-contents">{activeCourse.modules.map(module => <li key={module.id}><a href={buildLocation(language, courseKey, module.module_key)}>{module.title}</a>{quizFor(courseKey, module.module_key, language) && <> · <a className="course-quiz-link" href={buildLocation(language, courseKey, module.module_key, 'tests')}>{language === 'ru' ? 'Тесты' : 'Quizzes'}</a></>}{(['exercises','notebooks'] as const).map(kind => practiceFor(courseKey,module.module_key,language,kind) && <span key={kind}> · <a className="course-quiz-link" href={buildLocation(language,courseKey,module.module_key,kind)}>{kind === 'exercises' ? (language === 'ru' ? 'Упражнения' : 'Exercises') : (language === 'ru' ? 'Ноутбуки' : 'Notebooks')}</a></span>)}</li>)}</ol>
          <p><a href={activeCourse.stepik_url}>{t.stepik}</a> · <a href={leanpubUrls[activeCourse.course_key][language]}>{t.leanpub}</a></p>
        </main>
      ) : !activeCourse || !activeModule ? (
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
                <p className="about-audience">{t.aboutAudience}</p>
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
                  const matchesCourse = (entry: { courseKey: string; language: string }) => entry.courseKey === course.course_key && entry.language === language;
                  const quizCount = quizModules.filter(matchesCourse).reduce((sum, entry) => sum + entry.count, 0);
                  const exerciseCount = practiceModules.filter(entry => matchesCourse(entry) && entry.section === 'exercises').reduce((sum, entry) => sum + entry.count, 0);
                  const notebookCount = practiceModules.filter(entry => matchesCourse(entry) && entry.section === 'notebooks').reduce((sum, entry) => sum + entry.count, 0);
                  return (
                    <article className="course-card" key={course.course_key} data-accent={course.course_key}>
                      <div className="course-card-top">
                        <span className="course-number">{courseNumbers[course.course_key]}</span>
                        <BookOpen size={22} />
                      </div>
                      <h3><a href={buildLocation(language, course.course_key)}>{course.title}</a></h3>
                      <p>{course.description}</p>
                      <div className="course-access">
                        <section className="access-panel access-panel-open">
                          <p className="access-title">{t.courseContentTitle}</p>
                          <p className="access-description">{t.courseContentText}</p>
                          <dl className="course-stats">
                            <div>
                              <dt>{course.module_count}</dt>
                              <dd>{t.modules}</dd>
                            </div>
                            <div>
                              <dt>{coursePageCounts[course.course_key][language]}</dt>
                              <dd>{t.pages} {publication.book_editions[course.course_key]}</dd>
                            </div>
                            <div>
                              <dt>{imageCount}</dt>
                              <dd>{t.illustrations}</dd>
                            </div>
                          </dl>
                          <ul className="course-volume">
                            <li>{t.quizQuestions}: <strong>{quizCount}</strong></li>
                            <li>{t.exerciseItems}: <strong>{exerciseCount}</strong></li>
                            <li>{notebookCount ? <>{t.notebookFiles}: <strong>{notebookCount}</strong></> : t.pythonFiles}</li>
                          </ul>
                          <a className="course-open" href={buildLocation(language, course.course_key, course.modules[0].module_key)}>
                            <span>{t.studyHere}<small>{t.siteHint}</small></span>
                            <ArrowRight size={18} />
                          </a>
                          <a
                            className="course-stepik"
                            href={course.stepik_url}
                            target="_blank"
                            rel="noreferrer"
                          >
                            <span>{t.stepik}<small>{t.stepikHint}</small></span>
                            <ExternalLink size={15} />
                          </a>
                        </section>

                        <section className="access-panel access-panel-book">
                          <p className="access-title">{t.bookTitle}</p>
                          <p className="access-description">{t.bookText}</p>
                          <a
                            className="course-leanpub"
                            href={leanpubUrls[course.course_key][language]}
                            target="_blank"
                            rel="noreferrer"
                          >
                            {t.leanpub}
                            <ExternalLink size={15} />
                          </a>
                        </section>
                      </div>
                    </article>
                  );
                })}
            </div>
          </section>

          <aside className="boundary-note">
            <span>{t.boundaryLabel}</span>
            <p>{t.boundary}</p>
          </aside>
        </main>
      ) : (
        <main className={`reader-layout ${sidebarHidden ? "left-menu-hidden" : ""} ${tocHidden ? "right-menu-hidden" : ""}`}>
          <aside id="course-menu" aria-label={t.modulesMenu} className={`course-sidebar ${drawerOpen ? "is-open" : ""}`}>
            <div className="sidebar-head">
              <a className="back-home" href={buildLocation(language)}>
                <Home size={16} />
                {t.home}
              </a>
              <button className="drawer-close" onClick={() => setDrawerOpen(false)} aria-label={t.hideModules}>
                <X size={20} />
              </button>
              <span className="sidebar-course-number">{courseNumbers[activeCourse.course_key]}</span>
              <p className="sidebar-title"><a href={buildLocation(language, courseKey)}>{activeCourse.title}</a></p>
              <div className="sidebar-course-links">
                <a href={activeCourse.stepik_url} target="_blank" rel="noreferrer">
                  {t.stepik}
                  <ExternalLink size={14} />
                </a>
                <a
                  href={leanpubUrls[activeCourse.course_key][language]}
                  target="_blank"
                  rel="noreferrer"
                >
                  {t.leanpub}
                  <ExternalLink size={14} />
                </a>
              </div>
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
                  <a key={module.id} className={module.id === activeModule.id ? "active" : ""}
                    aria-current={module.id === activeModule.id ? "page" : undefined}
                    href={buildLocation(language, activeCourse.course_key, module.module_key, moduleSection(module.module_key))}>
                    <span>{formatModuleNumber(module)}</span>
                    <strong>{module.title.replace(/^(Module|Модуль)\s+\d+b?[.:]?\s*/i, "")}</strong>
                  </a>
                ))
              ) : (
                <p className="empty-search">{t.noResults}</p>
              )}
            </nav>
          </aside>
          {(drawerOpen || tocDrawerOpen) && (
            <button
              className={`drawer-scrim ${tocDrawerOpen ? "toc-scrim" : "module-scrim"}`}
              onClick={() => {
                setDrawerOpen(false);
                setTocDrawerOpen(false);
              }}
              aria-label={tocDrawerOpen ? t.hideOutline : t.hideModules}
            />
          )}

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

            <nav className="material-tabs" aria-label={language === 'ru' ? 'Материалы модуля' : 'Module materials'}>
              <a href={buildLocation(language, courseKey, moduleKey)} aria-current={!section ? 'page' : undefined}>{language === 'ru' ? 'Лекция' : 'Lecture'}</a>
              {activeQuiz && <a href={buildLocation(language, courseKey, moduleKey, 'tests')} aria-current={section === 'tests' ? 'page' : undefined}>{language === 'ru' ? 'Тесты' : 'Quizzes'} <span>{activeQuiz.count}</span></a>}
              {(['exercises', 'notebooks'] as const).map(kind => {
                const entry = practiceFor(courseKey, moduleKey, language, kind);
                return entry && <a key={kind} href={buildLocation(language, courseKey, moduleKey, kind)} aria-current={section === kind ? 'page' : undefined}>{kind === 'exercises' ? (language === 'ru' ? 'Упражнения' : 'Exercises') : (language === 'ru' ? 'Ноутбуки' : 'Notebooks')} <span>{entry.count}</span></a>;
              })}
            </nav>

            <StepikNotice language={language} />
            {loading && !markdown && <div className="lecture-state">{section ? (language === 'ru' ? 'Загружаю материалы…' : 'Loading materials…') : t.loading}</div>}
            {loadError && <div className="lecture-state error">{t.error}</div>}
            {quiz && activeQuiz && !loadError && <QuizReader key={`${courseKey}:${moduleKey}`} quiz={quiz} revision={activeQuiz.revision} />}
            {practice && activePractice && !loadError && <PracticeReader key={`${courseKey}:${moduleKey}:${section}`} page={practice} revision={activePractice.revision} />}
            {markdown && !loadError && !section && (
              <div className={`lecture-prose${courseKey === 'modern-llms' && moduleKey === 'module-20' ? ' lecture-prose-equation-cards' : ''}`} aria-busy={loading}>
                <ReactMarkdown
                  remarkPlugins={[remarkGfm, remarkMath]}
                  rehypePlugins={[rehypeKatex]}
                  components={{
                    span: ({ node, ...props }) => {
                      const equation = courseKey === 'modern-llms' && moduleKey === 'module-20' && props.className?.split(' ').includes('katex-display');
                      if (equation) return <span className="reference-equation" tabIndex={0} role="group" aria-label={language === 'ru' ? 'Формула' : 'Equation'} dangerouslySetInnerHTML={{ __html: module20Equation(node, language) }} />;
                      return <span {...props} />;
                    },
                    h2: ({ children, node }) => {
                      const Tag = `h${node?.position ? semanticHeadings.get(node.position.start.line) ?? 2 : 2}` as "h2" | "h3" | "h4";
                      return <Tag className="markdown-h2" id={node?.position ? headingIds.get(node.position.start.line) : undefined}>{children}</Tag>;
                    },
                    h3: ({ children, node }) => {
                      const Tag = `h${node?.position ? semanticHeadings.get(node.position.start.line) ?? 3 : 3}` as "h2" | "h3" | "h4";
                      return <Tag className="markdown-h3" id={node?.position ? headingIds.get(node.position.start.line) : undefined}>{children}</Tag>;
                    },
                    h4: ({ children, node }) => {
                      const Tag = `h${node?.position ? semanticHeadings.get(node.position.start.line) ?? 4 : 4}` as "h2" | "h3" | "h4";
                      return <Tag className="markdown-h4" id={node?.position ? headingIds.get(node.position.start.line) : undefined}>{children}</Tag>;
                    },
                    a: ({ href, children, ...props }) => {
                      const internalModule = href?.startsWith("/") ? null : findInternalModule(href);
                      if (internalModule && activeCourse.modules.some((module) => module.module_key === internalModule)) {
                        return (
                          <a
                            href={`${buildLocation(language, activeCourse.course_key, internalModule)}${href?.includes("#") ? `#${href.split("#")[1]}` : ""}`}
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
                    img: ({ src, alt, node }) => {
                      const value = String(src ?? "");
                      const resolved = /^https?:|^data:/.test(value)
                        ? value
                        : `${getBaseUrl()}${value.replace(/^\.\//, "")}`;
                      const section = headingIndex.filter(heading => heading.line < (node?.position?.start.line ?? 0)).at(-1)?.label ?? activeModule.title;
                      return <img src={resolved} alt={alt || `${language === "ru" ? "Иллюстрация к разделу" : "Illustration for"}: ${section}`} loading="lazy" />;
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
                <a href={buildLocation(language, activeCourse.course_key, previousModule.module_key, moduleSection(previousModule.module_key))}>
                  <ChevronLeft size={18} />
                  <span>
                    <small>{t.previous}</small>
                    {formatModuleNumber(previousModule)}
                  </span>
                </a>
              ) : (
                <span />
              )}
              {nextModule && (
                <a href={buildLocation(language, activeCourse.course_key, nextModule.module_key, moduleSection(nextModule.module_key))}>
                  <span>
                    <small>{t.next}</small>
                    {formatModuleNumber(nextModule)}
                  </span>
                  <ChevronRight size={18} />
                </a>
              )}
            </nav>
          </article>

          <aside id="lecture-outline" aria-label={t.outlineMenu} className={`toc-rail ${tocDrawerOpen ? "is-open" : ""}`}>
            <button className="drawer-close" onClick={() => setTocDrawerOpen(false)} aria-label={t.hideOutline}>
              <X size={20} />
            </button>
            <p>{t.contents}</p>
            <nav>
              {headings.map((heading) => (
                <a key={heading.id} href={`#${heading.id}`} onClick={() => setTocDrawerOpen(false)}>
                  {practice ? <PracticeMarkdown ru={language === 'ru'} inline>{heading.label}</PracticeMarkdown> : section === 'tests' ? <QuizMarkdown inline>{heading.label}</QuizMarkdown> : heading.label.replace(/^\d+[.:]\s*/, "")}
                </a>
              ))}
            </nav>
          </aside>
        </main>
      )}

      <footer className="site-footer">
        <span>
          © 2026 {t.author} · {language === "ru" ? "Тексты обновлены" : "Text updated"} {publication.content_updated}
        </span>
        <nav aria-label="Publication information">
          <a
            href="https://github.com/VladGKulikov/VladGKulikov.github.io/blob/main/LICENSE.md"
            target="_blank"
            rel="noreferrer"
          >
            {t.licenses}
            <ExternalLink size={13} />
          </a>
          <a
            href="/CITATION.cff"
            target="_blank"
            rel="noreferrer"
          >
            {t.citation}
            <ExternalLink size={13} />
          </a>
          <a
            href={`https://doi.org/${publication.version_doi}`}
            target="_blank"
            rel="noreferrer"
          >
            DOI {publication.edition}
            <ExternalLink size={13} />
          </a>
          <a href="https://github.com/VladGKulikov/VladGKulikov.github.io" target="_blank" rel="noreferrer">
            {t.openSource}
            <ExternalLink size={13} />
          </a>
        </nav>
      </footer>
    </div>
  );
}
