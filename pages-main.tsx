import { StrictMode } from "react";
import { createRoot, hydrateRoot } from "react-dom/client";
import "katex/dist/katex.min.css";
import "./app/globals.css";
import { CourseLibrary } from "./app/CourseLibrary";
import { buildLocation, parseUrl, type PageRoute } from "./app/site";
import { campaignQuery } from "./app/analytics";

const root = document.getElementById("root")!;
const data = document.getElementById("page-data")?.textContent;
const initial = data ? JSON.parse(data) as { initialPage: PageRoute; initialMarkdown: string } : undefined;
const route = parseUrl(new URL(window.location.href));
// GitHub Pages cannot redirect query selections at the HTTP layer. Preserve
// old links and section anchors. Canonicalize in place so external referrers
// and safe UTM attribution survive query links without a second page load.
if (initial && !initial.initialPage.notFound && route.notFound) {
  window.location.replace("/404.html");
} else {
  const legacy = initial && !initial.initialPage.notFound && !route.notFound &&
    buildLocation(route.language, route.courseKey, route.moduleKey, route.section) !== window.location.pathname;
  if (legacy) window.history.replaceState(window.history.state, "", `${buildLocation(route.language, route.courseKey, route.moduleKey, route.section)}${campaignQuery(window.location.search)}${window.location.hash}`);
  const props = legacy ? { initialPage: route, initialMarkdown: "" } : initial;
  const app = <StrictMode><CourseLibrary {...props} /></StrictMode>;
  if (initial && !legacy) hydrateRoot(root, app);
  else createRoot(root).render(app);
}
