import { renderToString } from "react-dom/server";
import { CourseLibrary } from "./app/CourseLibrary";
import { courses, buildLocation, pageMetadata, type PageRoute } from "./app/site";
import { quizFor } from './app/quizzes';
import { practiceFor } from './app/practice';

export const routes: PageRoute[] = (["en", "ru"] as const).flatMap(language => [
  { language, courseKey: null, moduleKey: null },
  ...courses.filter(course => course.language === language).flatMap(course => [
    { language, courseKey: course.course_key, moduleKey: null },
    ...course.modules.flatMap(module => [
      { language, courseKey: course.course_key, moduleKey: module.module_key },
      ...(quizFor(course.course_key, module.module_key, language) ? [{ language, courseKey: course.course_key, moduleKey: module.module_key, section: 'tests' as const }] : []),
      ...(['exercises','notebooks'] as const).flatMap(section => practiceFor(course.course_key,module.module_key,language,section) ? [{language,courseKey:course.course_key,moduleKey:module.module_key,section}] : []),
    ]),
  ]),
]);
export { courses, buildLocation, pageMetadata };
export { parseUrl } from "./app/site";
export { progressModuleKey } from './app/module-progress';
export { quizModules, gradeQuestion, parseNumericAnswer } from './app/quizzes';
export { practiceModules } from './app/practice';
export { highlightPython } from './app/CodeBlock';
export function renderPage(initialPage: PageRoute, initialMarkdown: string) {
  return renderToString(<CourseLibrary initialPage={initialPage} initialMarkdown={initialMarkdown} />);
}
