import { renderToString } from "react-dom/server";
import { CourseLibrary } from "./app/CourseLibrary";
import { courses, buildLocation, pageMetadata, type PageRoute } from "./app/site";
import { quizFor } from './app/quizzes';

export const routes: PageRoute[] = (["en", "ru"] as const).flatMap(language => [
  { language, courseKey: null, moduleKey: null },
  ...courses.filter(course => course.language === language).flatMap(course => [
    { language, courseKey: course.course_key, moduleKey: null },
    ...course.modules.flatMap(module => [
      { language, courseKey: course.course_key, moduleKey: module.module_key },
      ...(quizFor(course.course_key, module.module_key, language) ? [{ language, courseKey: course.course_key, moduleKey: module.module_key, section: 'tests' as const }] : []),
    ]),
  ]),
]);
export { courses, buildLocation, pageMetadata };
export { parseUrl } from "./app/site";
export { quizModules, gradeQuestion, parseNumericAnswer } from './app/quizzes';
export function renderPage(initialPage: PageRoute, initialMarkdown: string) {
  return renderToString(<CourseLibrary initialPage={initialPage} initialMarkdown={initialMarkdown} />);
}
