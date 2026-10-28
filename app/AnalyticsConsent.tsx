"use client";

import { useState, useSyncExternalStore } from "react";
import { analyticsEnabled, getAnalyticsConsent, setAnalyticsConsent, subscribeAnalyticsConsent } from "./analytics";
import type { Language } from "./site";

const serverConsent = () => null;

export function AnalyticsConsent({ language }: { language: Language }) {
  const consent = useSyncExternalStore(subscribeAnalyticsConsent, getAnalyticsConsent, serverConsent);
  const [open, setOpen] = useState(false);
  const ru = language === "ru";
  if (!analyticsEnabled) return null;
  const choose = (value: "granted" | "denied") => {
    setOpen(false);
    setAnalyticsConsent(value);
  };
  return <div className="analytics-controls js-only">
    <button type="button" className="privacy-settings" aria-expanded={open || consent === null} aria-controls="analytics-choice" onClick={() => setOpen(!open)}>
      {ru ? "Приватность и аналитика" : "Privacy & analytics"}
    </button>
    {(open || consent === null) && <aside id="analytics-choice" className="analytics-choice" aria-label={ru ? "Согласие на аналитику" : "Analytics consent"}>
      <strong>{ru ? "Необязательная аналитика" : "Optional analytics"}</strong>
      <p>{ru
        ? "С вашего согласия Google Analytics 4 использует cookies, чтобы измерять просмотры страниц, источники и UTM-кампании, а также переходы на Stepik и Leanpub. Google обрабатывает технические данные браузера и подключения; обработка может происходить за пределами вашей страны. Рекламная персонализация отключена."
        : "With your consent, Google Analytics 4 uses cookies to measure page views, referral sources and UTM campaigns, and clicks to Stepik and Leanpub. Google processes browser and connection information; processing may take place outside your country. Advertising personalization is disabled."}</p>
      <p>{ru
        ? "До согласия GA4 не загружается. Ответы, прогресс и поиск по модулям не передаются. Выбор хранится в этом браузере до 180 дней; cookies аналитики — до 180 дней. Отказ не влияет на курсы. Изменить выбор можно здесь в любое время; отзыв согласия удаляет cookies GA и перезагружает страницу. Сигналы GPC / Do Not Track означают отказ."
        : "GA4 stays unloaded until you consent. Answers, progress and module searches are not sent. Your choice is stored in this browser for up to 180 days; analytics cookies last up to 180 days. Declining does not affect the courses. Change your choice here anytime; withdrawal deletes GA cookies and reloads the page. GPC / Do Not Track signals mean no analytics."}</p>
      <p><a href="https://policies.google.com/privacy" target="_blank" rel="noreferrer">{ru ? "Конфиденциальность Google" : "Google privacy policy"}</a>{" · "}<a href="https://policies.google.com/technologies/partner-sites" target="_blank" rel="noreferrer">{ru ? "Использование данных" : "How Google uses data"}</a></p>
      <p role="status">{ru ? "Текущий выбор: " : "Current choice: "}{consent === "granted" ? (ru ? "разрешено" : "allowed") : consent === "denied" ? (ru ? "отказ" : "declined") : (ru ? "не выбран" : "not set")}</p>
      <div className="analytics-actions">
        <button type="button" onClick={() => choose("denied")}>{ru ? "Отклонить" : "Decline"}</button>
        <button type="button" onClick={() => choose("granted")}>{ru ? "Разрешить аналитику" : "Allow analytics"}</button>
        {consent !== null && <button type="button" onClick={() => setOpen(false)}>{ru ? "Закрыть" : "Close"}</button>}
      </div>
    </aside>}
  </div>;
}
