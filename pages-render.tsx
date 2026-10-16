import { renderToString } from "react-dom/server";
import { CourseLibrary } from "./app/CourseLibrary";
import { courses, buildLocation, pageMetadata, type PageRoute } from "./app/site";

export const routes: PageRoute[] = (["en", "ru"] as const).flatMap(language => [
  { language, courseKey: null, moduleKey: null },
  ...courses.filter(course => course.language === language).flatMap(course => [
    { language, courseKey: course.course_key, moduleKey: null },
    ...course.modules.map(module => ({ language, courseKey: course.course_key, moduleKey: module.module_key })),
  ]),
]);
export { courses, buildLocation, pageMetadata };
export { parseUrl } from "./app/site";
export function renderPage(initialPage: PageRoute, initialMarkdown: string) {
  return renderToString(<CourseLibrary initialPage={initialPage} initialMarkdown={initialMarkdown} />);
}
