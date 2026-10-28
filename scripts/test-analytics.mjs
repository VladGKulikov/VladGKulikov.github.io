import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { test } from 'node:test';
import ts from 'typescript';

const origin = 'https://vladkulikov.com';
const home = { language: 'en', courseKey: null, moduleKey: null };
const lecture = { language: 'en', courseKey: 'modern-llms', moduleKey: 'module-01' };
const key = 'analytics-consent-v1';
const saved = value => JSON.stringify({ value, expires: Date.now() + 86400000 });

// Execute the real TS module and its real route/catalog dependencies in fresh
// browser contexts. No analytics requests or production test events are sent.
function browser({ search = '', referrer = '', consent, blockedStorage = false, failedWrites = false, gpc = false, dnt = '' } = {}) {
  const storage = new Map(consent === undefined ? [] : [[key, consent]]);
  const scripts = [], windowEvents = new Map(), documentEvents = new Map(), cookies = new Map();
  let reloads = 0;
  const win = {
    location: { origin, hostname: 'vladkulikov.com', search, reload() { reloads++; } },
    addEventListener(name, listener) { windowEvents.set(name, listener); },
    removeEventListener(name) { windowEvents.delete(name); },
  };
  const doc = {
    referrer,
    head: { append(script) { scripts.push(script); } },
    createElement() { return {}; },
    addEventListener(name, listener) { documentEvents.set(name, listener); },
    get cookie() { return [...cookies].map(([k, v]) => `${k}=${v}`).join('; '); },
    set cookie(value) {
      const [pair] = value.split(';');
      const [name, val] = pair.split('=');
      if (value.includes('Max-Age=0')) cookies.delete(name); else cookies.set(name, val);
    },
  };
  const localStorage = {
    getItem(k) { if (blockedStorage) throw Error('blocked'); return storage.get(k) ?? null; },
    setItem(k, v) { if (blockedStorage || failedWrites) throw Error('blocked'); storage.set(k, v); },
    removeItem(k) { if (blockedStorage) throw Error('blocked'); storage.delete(k); },
  };
  const cache = new Map();
  function load(filename) {
    if (filename.endsWith('.json')) return JSON.parse(fs.readFileSync(filename, 'utf8'));
    if (cache.has(filename)) return cache.get(filename).exports;
    const mod = { exports: {} };
    cache.set(filename, mod);
    const source = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
      compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
    }).outputText;
    vm.runInNewContext(source, {
      module: mod, exports: mod.exports, URL, URLSearchParams, Date,
      window: win, document: doc, localStorage,
      navigator: { globalPrivacyControl: gpc, doNotTrack: dnt },
      require(specifier) {
        const target = path.resolve(path.dirname(filename), specifier);
        return load(path.extname(target) ? target : `${target}.ts`);
      },
    }, { filename });
    return mod.exports;
  }
  const analytics = load(path.resolve('app/analytics.ts'));
  const events = name => (win.dataLayer ?? []).map(args => Array.from(args)).filter(args => args[0] === 'event' && args[1] === name).map(args => args[2]);
  const click = (href, type = 'click', button = 0) => documentEvents.get(type)?.({ type, button, target: { closest() { return { href }; } } });
  return { analytics, win, storage, scripts, doc, cookies, windowEvents, events, click, reloads: () => reloads };
}

test('config enables exactly the requested stream', () => {
  const config = JSON.parse(fs.readFileSync('SITE_CONFIG.json', 'utf8'));
  assert.deepEqual(config.analytics, { provider: 'ga4', measurementId: 'G-MTQTZTSE97' });
});

test('no tag, queue, events or cookies before consent or after rejection', () => {
  for (const consent of [undefined, saved('denied'), '{bad json', saved('invalid'), JSON.stringify({ value: 'granted', expires: 1 })]) {
    const b = browser({ consent });
    b.analytics.trackPage(home);
    b.analytics.trackPage(lecture);
    b.analytics.setAnalyticsConsent('denied');
    assert.equal(b.scripts.length, 0);
    assert.equal(b.win.dataLayer, undefined);
    assert.equal(b.doc.cookie, '');
  }
});

test('initial view preserves safe UTM and external source, with one tag/config', () => {
  const b = browser({ consent: saved('granted'), referrer: 'https://www.google.com/search?q=private#secret', search: '?utm_source=newsletter&utm_medium=email&utm_campaign=AI_2026&utm_content=hero&email=private@example.com&token=secret&gclid=secret' });
  b.analytics.trackPage(home);
  b.analytics.trackPage(home); // React StrictMode / effect rerun.
  assert.equal(b.scripts.length, 1);
  assert.equal(b.scripts[0].src, 'https://www.googletagmanager.com/gtag/js?id=G-MTQTZTSE97');
  assert.equal(b.scripts[0].referrerPolicy, 'origin');
  const view = b.events('page_view');
  assert.equal(view.length, 1);
  assert.equal(view[0].page_location, `${origin}/?utm_source=newsletter&utm_medium=email&utm_campaign=AI_2026&utm_content=hero`);
  assert.equal(view[0].page_referrer, 'https://www.google.com/');
  const calls = b.win.dataLayer.map(args => Array.from(args));
  assert.equal(calls[0][0], 'consent');
  assert.equal(calls[0][2].analytics_storage, 'denied');
  assert.equal(calls[1][2].analytics_storage, 'granted');
  assert.equal(calls[1][2].ad_storage, 'denied');
  const config = calls.find(args => args[0] === 'config')[2];
  assert.equal(config.send_page_view, false);
  assert.equal(config.allow_google_signals, false);
  assert.equal(config.allow_ad_personalization_signals, false);
  assert.equal(config.cookie_update, false);
  assert(!JSON.stringify(calls).includes('secret'));
  assert(!JSON.stringify(calls).includes('private'));
});

test('SPA language, sections and Back/Forward have one view with previous canonical referrer', () => {
  const b = browser({ consent: saved('granted') });
  const routes = [home, lecture, { ...lecture, language: 'ru' }, { ...lecture, language: 'ru', section: 'tests' }, lecture, home, lecture];
  for (const route of routes) { b.analytics.trackPage(route); b.analytics.trackPage(route); }
  const views = b.events('page_view');
  assert.equal(views.length, routes.length);
  assert.equal(views[0].page_referrer, '');
  for (let i = 1; i < views.length; i++) assert.equal(views[i].page_referrer, views[i - 1].page_location);
  assert.equal(views[2].page_location, `${origin}/ru/courses/modern-llms/module-01/`);
  assert(views[2].page_title.includes('Модуль 1.'));
  assert(views[3].page_location.endsWith('/tests/'));
  assert.equal(b.scripts.length, 1);
});

test('delayed opt-in preserves landing attribution without replaying earlier pages', () => {
  const b = browser({ referrer: 'https://t.me/s/private?user=secret', search: '?utm_source=telegram&utm_medium=social' });
  b.analytics.trackPage(home);
  b.win.location.search = '';
  b.analytics.trackPage(lecture);
  b.analytics.setAnalyticsConsent('granted');
  b.analytics.setAnalyticsConsent('granted');
  const views = b.events('page_view');
  assert.equal(views.length, 1);
  assert.equal(views[0].page_location, `${origin}/courses/modern-llms/module-01/?utm_source=telegram&utm_medium=social`);
  assert.equal(views[0].page_referrer, 'https://t.me/');
});

test('campaign allowlist rejects emails, URLs, long IDs and arbitrary query keys', () => {
  const { analytics: a } = browser();
  assert.equal(a.campaignQuery('?utm_source=mail%40example.com&utm_medium=https%3A%2F%2Fprivate&utm_id=123456789&email=person&query=secret'), '');
  assert.equal(a.campaignQuery('?utm_campaign=%D0%9A%D1%83%D1%80%D1%81%20LLM&utm_source=Telegram&token=secret'), '?utm_source=Telegram&utm_campaign=%D0%9A%D1%83%D1%80%D1%81+LLM');
  assert.equal(a.campaignQuery(`?utm_source=${'a'.repeat(101)}`), '');
});

test('full-page same-site referrer is a known canonical route without queries or hashes', () => {
  const b = browser({ consent: saved('granted'), referrer: `${origin}/ru/courses/modern-llms/module-01/?email=private#answer` });
  b.analytics.trackPage(home);
  assert.equal(b.events('page_view')[0].page_referrer, `${origin}/ru/courses/modern-llms/module-01/`);
});

test('Stepik/Leanpub outbound clicks are once per activation and strip query/hash', () => {
  const b = browser({ consent: saved('granted') });
  b.analytics.trackPage(home); b.analytics.trackPage(home);
  b.click('https://stepik.org/297276?email=private#answer');
  b.click('https://leanpub.com/modern-llms?token=secret', 'auxclick', 1);
  b.click('https://stepik.org/297276', 'auxclick', 2);
  b.click('https://stepik.org.evil.example/');
  b.click('https://example.com/');
  const clicks = b.events('outbound_click');
  assert.equal(clicks.length, 2);
  assert.equal(clicks[0].link_url, 'https://stepik.org/297276');
  assert.equal(clicks[1].link_url, 'https://leanpub.com/modern-llms');
  assert.equal(clicks[0].page_location, `${origin}/`);
});

test('withdrawal disables GA, clears only GA cookies and reloads with persisted denial', () => {
  const b = browser({ consent: saved('granted') });
  b.analytics.trackPage(home);
  b.cookies.set('_ga', 'client'); b.cookies.set('_ga_MTQTZTSE97', 'session'); b.cookies.set('course-progress', 'keep');
  b.analytics.setAnalyticsConsent('denied');
  assert.equal(b.win['ga-disable-G-MTQTZTSE97'], true);
  assert.equal(b.reloads(), 1);
  assert.equal(b.doc.cookie, 'course-progress=keep');
  assert.equal(JSON.parse(b.storage.get(key)).value, 'denied');
  b.click('https://stepik.org/297276');
  assert.equal(b.events('outbound_click').length, 0);
  const fresh = browser({ consent: b.storage.get(key) });
  fresh.analytics.trackPage(home);
  assert.equal(fresh.scripts.length, 0);
});

test('cross-tab withdrawal and clearing storage stop an already loaded tag', () => {
  for (const clearedKey of [key, null]) {
    const b = browser({ consent: saved('granted') });
    const unsubscribe = b.analytics.subscribeAnalyticsConsent(() => {});
    b.analytics.trackPage(home);
    b.storage.delete(key);
    b.windowEvents.get('storage')({ key: clearedKey });
    assert.equal(b.reloads(), 1);
    assert.equal(b.win['ga-disable-G-MTQTZTSE97'], true);
    unsubscribe();
    assert.equal(b.windowEvents.has('storage'), false);
  }
});

test('blocked storage and quota failures allow memory-only opt-in and safe withdrawal', () => {
  for (const options of [{ blockedStorage: true }, { failedWrites: true, consent: saved('granted') }]) {
    const b = browser(options);
    b.analytics.trackPage(home);
    b.analytics.setAnalyticsConsent('granted');
    assert.equal(b.scripts.length, 1);
    b.analytics.setAnalyticsConsent('denied');
    assert.equal(b.analytics.getAnalyticsConsent(), 'denied');
    assert.equal(b.reloads(), 1);
    assert.equal(b.storage.has(key), false);
  }
});

test('GPC and Do Not Track block even saved grants and attempts to opt in', () => {
  for (const options of [{ gpc: true }, { dnt: '1' }]) {
    const b = browser({ ...options, consent: saved('granted') });
    b.analytics.trackPage(home);
    b.analytics.setAnalyticsConsent('granted');
    assert.equal(b.scripts.length, 0);
    assert.equal(b.analytics.getAnalyticsConsent(), 'denied');
  }
});

test('404 routes do not create fabricated home page views or outbound events', () => {
  const b = browser({ consent: saved('granted') });
  b.analytics.trackPage({ ...home, notFound: true });
  assert.equal(b.scripts.length, 0);
  b.analytics.trackPage(home);
  b.analytics.trackPage({ ...home, notFound: true });
  b.click('https://stepik.org/297276');
  assert.equal(b.events('page_view').length, 1);
  assert.equal(b.events('outbound_click').length, 0);
});
