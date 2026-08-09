import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "katex/dist/katex.min.css";
import "./app/globals.css";
import { CourseLibrary } from "./app/CourseLibrary";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <CourseLibrary />
  </StrictMode>,
);
