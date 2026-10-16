import config from "../SITE_CONFIG.json";
import { pageMetadata, siteOrigin, type PageRoute } from "./site";

let installed = false;
let previousPath = "";
const enabled = config.analytics.provider === "ga4" && /^G-[A-Z0-9]+$/.test(config.analytics.measurementId ?? "");
type AnalyticsWindow = Window & { dataLayer?: unknown[]; gtag?: (...args: unknown[]) => void };

export function trackPage(route: PageRoute) {
  if (!enabled || route.notFound) return;
  const win = window as AnalyticsWindow;
  const id = config.analytics.measurementId!;
  if (!installed) {
    installed = true;
    win.dataLayer = win.dataLayer ?? [];
    // gtag's official queue contract uses an Arguments object, rather than an Array.
    // eslint-disable-next-line prefer-rest-params
    win.gtag = function () { win.dataLayer!.push(arguments); };
    win.gtag("js", new Date());
    win.gtag("config", id, { send_page_view: false, allow_google_signals: false, allow_ad_personalization_signals: false });
    const script = document.createElement("script");
    script.async = true;
    script.src = `https://www.googletagmanager.com/gtag/js?id=${encodeURIComponent(id)}`;
    document.head.append(script);
    document.addEventListener("click", event => {
      const anchor = (event.target as Element)?.closest?.("a[href]") as HTMLAnchorElement | null;
      if (!anchor) return;
      const target = new URL(anchor.href);
      if (!["stepik.org", "leanpub.com"].includes(target.hostname)) return;
      win.gtag?.("event", "outbound_click", { destination: target.hostname, link_url: `${target.origin}${target.pathname}`, page_location: `${siteOrigin}${previousPath}` });
    });
  }
  const meta = pageMetadata(route);
  if (previousPath === meta.path) return;
  const referrer = previousPath ? `${siteOrigin}${previousPath}` : "";
  previousPath = meta.path;
  win.gtag?.("event", "page_view", { page_location: meta.canonical, page_title: meta.title, page_referrer: referrer });
}
