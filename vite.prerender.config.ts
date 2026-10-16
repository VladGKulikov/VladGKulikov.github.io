import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  publicDir: false,
  build: { ssr: "pages-render.tsx", outDir: "build-static", emptyOutDir: true },
});
