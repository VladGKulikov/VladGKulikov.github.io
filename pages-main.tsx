import { StrictMode } from "react";
import { createRoot, hydrateRoot } from "react-dom/client";
import "katex/dist/katex.min.css";
import "./app/globals.css";
import { CourseLibrary } from "./app/CourseLibrary";
import { buildLocation, parseUrl, type PageRoute } from "./app/site";

const root = document.getElementById("root")!;
const data = document.getElementById("page-data")?.textContent;
const initial = data ? JSON.parse(data) as { initialPage: PageRoute; initialMarkdown: string } : undefined;
const route = parseUrl(new URL(window.location.href));
// GitHub Pages cannot redirect query selections at the HTTP layer. Preserve
// old links and section anchors, then hydrate the permanent destination page.
if (initial && !initial.initialPage.notFound && route.notFound) {
  window.location.replace("/404.html");
} else if (initial && !initial.initialPage.notFound && !route.notFound &&
    buildLocation(route.language, route.courseKey, route.moduleKey) !== window.location.pathname) {
  window.location.replace(`${buildLocation(route.language, route.courseKey, route.moduleKey)}${window.location.hash}`);
} else {
  const app = <StrictMode><CourseLibrary {...initial} /></StrictMode>;
  if (initial) hydrateRoot(root, app);
  else createRoot(root).render(app);
}
