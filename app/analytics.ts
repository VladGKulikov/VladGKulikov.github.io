import config from "../SITE_CONFIG.json";
import { pageMetadata, parseUrl, type PageRoute } from "./site";

export const analyticsEnabled = config.analytics.provider === "ga4" && /^G-[A-Z0-9]+$/.test(config.analytics.measurementId ?? "");
export const consentStorageKey = "analytics-consent-v1";
const consentLifetime = 180 * 24 * 60 * 60 * 1000;
type Consent = "granted" | "denied" | null;
type Choice = { value: Exclude<Consent, null>; expires: number };
type AnalyticsWindow = Window & { dataLayer?: unknown[]; gtag?: (...args: unknown[]) => void; [key: `ga-disable-${string}`]: boolean | undefined };
let memoryChoice: Choice | null = null;
let installed = false;
let currentRoute: PageRoute | null = null;
let previousPath = "";
let previousLocation = "";
let landing: { campaign: string; referrer: string } | null = null;
const listeners = new Set<() => void>();

// Campaign labels are public marketing metadata, never names, emails or IDs.
// Exclude click identifiers, arbitrary query keys and hashes.
export function campaignQuery(search: string) {
  const input = new URLSearchParams(search);
  const output = new URLSearchParams();
  for (const key of ["utm_id", "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "utm_source_platform", "utm_creative_format", "utm_marketing_tactic"]) {
    const value = input.get(key);
    if (value && /^[\p{L}\p{N} _.-]{1,100}$/u.test(value) && !/\d{7,}/.test(value)) output.set(key, value);
  }
  return output.size ? `?${output}` : "";
}

function externalReferrer() {
  try {
    const url = new URL(document.referrer);
    if (url.origin === window.location.origin || url.origin === config.origin) {
      const route = parseUrl(url);
      if (!route.notFound) return pageMetadata(route).canonical;
    }
    return /^https?:$/.test(url.protocol) ? `${url.origin}/` : "";
  } catch { return ""; }
}

export function privacySignal() {
  return (navigator as Navigator & { globalPrivacyControl?: boolean }).globalPrivacyControl === true || navigator.doNotTrack === "1";
}

export function getAnalyticsConsent(): Consent {
  if (typeof window === "undefined") return null;
  if (privacySignal()) return "denied";
  let choice = memoryChoice;
  if (!choice) {
    try { choice = JSON.parse(localStorage.getItem(consentStorageKey) ?? "null"); } catch { /* Memory-only choice if storage is unavailable. */ }
  }
  return choice && (choice.value === "granted" || choice.value === "denied") && choice.expires > Date.now() && choice.expires <= Date.now() + consentLifetime ? choice.value : null;
}

function clearAnalyticsCookies() {
  const domains = window.location.hostname.split(".");
  for (const item of document.cookie.split(";")) {
    const name = item.split("=")[0].trim();
    if (name !== "_ga" && !name.startsWith("_ga_")) continue;
    const cookie = `${name}=; Max-Age=0; Path=/; SameSite=Lax`;
    document.cookie = cookie;
    for (let i = 0; i < domains.length - 1; i++) {
      const domain = domains.slice(i).join(".");
      document.cookie = `${cookie}; Domain=${domain}`;
      document.cookie = `${cookie}; Domain=.${domain}`;
    }
  }
}

function stopAnalytics() {
  const win = window as unknown as AnalyticsWindow;
  win[`ga-disable-${config.analytics.measurementId}`] = true;
  clearAnalyticsCookies();
  // Removing a script cannot unload its timers/listeners. Reload with the saved
  // denial to remove the tag entirely, without sending cookieless denial pings.
  if (installed) {
    win.dataLayer = [];
    win.gtag = () => {};
    window.location.reload();
  }
}

export function setAnalyticsConsent(value: Exclude<Consent, null>) {
  memoryChoice = { value: privacySignal() ? "denied" : value, expires: Date.now() + consentLifetime };
  try { localStorage.setItem(consentStorageKey, JSON.stringify(memoryChoice)); } catch {
    // A failed write must not leave a previous persisted grant after withdrawal.
    try { localStorage.removeItem(consentStorageKey); } catch { /* Storage unavailable. */ }
  }
  for (const listener of listeners) listener();
  if (getAnalyticsConsent() === "granted") {
    if (currentRoute) trackPage(currentRoute);
  } else stopAnalytics();
}

function onStorage(event: StorageEvent) {
  if (event.key !== consentStorageKey && event.key !== null) return;
  memoryChoice = null;
  for (const listener of listeners) listener();
  if (getAnalyticsConsent() !== "granted") stopAnalytics();
  else if (currentRoute) trackPage(currentRoute);
}

export function subscribeAnalyticsConsent(listener: () => void) {
  if (!listeners.size) window.addEventListener("storage", onStorage);
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
    if (!listeners.size) window.removeEventListener("storage", onStorage);
  };
}

function installAnalytics(page: { page_location: string; page_title: string; page_referrer: string }) {
  installed = true;
  const win = window as unknown as AnalyticsWindow;
  win[`ga-disable-${config.analytics.measurementId}`] = false;
  win.dataLayer = win.dataLayer ?? [];
  // gtag's official queue contract uses Arguments objects, rather than Arrays.
  // eslint-disable-next-line prefer-rest-params
  win.gtag = function () { win.dataLayer!.push(arguments); };
  // Basic Consent Mode: no tag, cookies or cookieless pings before opt-in.
  const denied = { analytics_storage: "denied", ad_storage: "denied", ad_user_data: "denied", ad_personalization: "denied" };
  win.gtag("consent", "default", denied);
  win.gtag("consent", "update", { ...denied, analytics_storage: "granted" });
  win.gtag("set", { ...page, ads_data_redaction: true, url_passthrough: false });
  win.gtag("js", new Date());
  // Enhanced Measurement is OFF in the web stream as well: send_page_view alone
  // does not disable history-based pageviews, search, forms or outbound capture.
  win.gtag("config", config.analytics.measurementId, { send_page_view: false, allow_google_signals: false, allow_ad_personalization_signals: false, cookie_expires: consentLifetime / 1000, cookie_update: false });
  const script = document.createElement("script");
  script.async = true;
  script.referrerPolicy = "origin";
  script.src = `https://www.googletagmanager.com/gtag/js?id=${encodeURIComponent(config.analytics.measurementId!)}`;
  document.head.append(script);
  const outbound = (event: MouseEvent) => {
    if (getAnalyticsConsent() !== "granted" || !currentRoute || currentRoute.notFound || (event.type === "auxclick" && event.button !== 1)) return;
    const anchor = (event.target as Element)?.closest?.("a[href]") as HTMLAnchorElement | null;
    if (!anchor) return;
    const target = new URL(anchor.href);
    if (!/^https?:$/.test(target.protocol) || !["stepik.org", "leanpub.com"].includes(target.hostname)) return;
    win.gtag?.("event", "outbound_click", { destination: target.hostname, link_url: `${target.origin}${target.pathname}`, page_location: previousLocation });
  };
  document.addEventListener("click", outbound);
  document.addEventListener("auxclick", outbound);
}

export function trackPage(route: PageRoute) {
  currentRoute = route;
  if (!analyticsEnabled || route.notFound) return;
  // Keep only sanitized attribution in memory while the banner is open.
  // Do not replay pre-consent pageviews when consent is later granted.
  landing ??= { campaign: campaignQuery(window.location.search), referrer: externalReferrer() };
  if (getAnalyticsConsent() !== "granted") {
    if (installed) stopAnalytics();
    return;
  }
  const meta = pageMetadata(route);
  if (previousPath === meta.path) return; // StrictMode, rerenders and hash changes.
  const page = { page_location: meta.canonical + (previousPath ? "" : landing.campaign), page_title: meta.title, page_referrer: previousPath ? previousLocation : landing.referrer };
  if (!installed) installAnalytics(page);
  // Keep automatic session/engagement events on the same sanitized page context.
  (window as unknown as AnalyticsWindow).gtag?.("set", page);
  (window as unknown as AnalyticsWindow).gtag?.("event", "page_view", page);
  previousPath = meta.path;
  previousLocation = meta.canonical;
}
