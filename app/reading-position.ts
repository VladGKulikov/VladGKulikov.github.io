export type ReadingPosition = {
  sectionKey: string;
  sectionIndex: number;
  blockKind: string;
  blockOrdinal: number;
  blockCount: number;
  blockProgress: number;
  sectionProgress: number;
  headerProgress: number | null;
};

type ReadingSection = { key: string; blocks: HTMLElement[] };

const clamp = (value: number) => Math.max(0, Math.min(1, value));
const readingInset = () => (document.querySelector(".topbar")?.getBoundingClientRect().height ?? 72) + 32;
const blockKind = (block: HTMLElement) => block.querySelector("img") ? "IMAGE" : block.tagName;

function readingSections(): ReadingSection[] {
  const prose = document.querySelector(".lecture-prose");
  if (!prose) return [];
  if (prose.hasAttribute('data-quiz')) {
    return Array.from(prose.querySelectorAll<HTMLElement>(':scope > [data-question-id]')).map(question => ({
      key: `question:${question.dataset.questionId}`,
      blocks: Array.from(question.children).filter(child => child.tagName !== 'NOSCRIPT') as HTMLElement[],
    }));
  }
  const sections: ReadingSection[] = [{ key: "preamble", blocks: [] }];
  for (const block of Array.from(prose.children) as HTMLElement[]) {
    if (block.tagName === "H2" || block.tagName === "H3") {
      // Section numbers are shared by the translations; heading slugs are not.
      const number = (block.textContent ?? "").match(/^(?:(?:Lesson|Урок)\s+)?(\d+(?:\.\d+)*)(?:[.:]|\s|$)/i)?.[1];
      sections.push({ key: number ? `number:${number}` : `index:${sections.length}`, blocks: [] });
    }
    sections[sections.length - 1].blocks.push(block);
  }
  return sections.filter((section) => section.blocks.length);
}

export function captureReadingPosition(): ReadingPosition | null {
  const sections = readingSections();
  if (!sections.length) return null;
  const inset = readingInset();
  // Layout coordinates can round to opposite sides of the reading line at
  // fractional display scales. Treat a heading within one CSS pixel as reached.
  const reached = inset + 1;
  let sectionIndex = 0;
  for (let index = 0; index < sections.length; index++) {
    if (sections[index].blocks[0].getBoundingClientRect().top <= reached) sectionIndex = index;
  }
  const section = sections[sectionIndex];
  let blockIndex = 0;
  for (let index = 0; index < section.blocks.length; index++) {
    if (section.blocks[index].getBoundingClientRect().top <= reached) blockIndex = index;
  }
  const block = section.blocks[blockIndex];
  const rect = block.getBoundingClientRect();
  const start = section.blocks[0].getBoundingClientRect().top;
  const last = section.blocks[section.blocks.length - 1].getBoundingClientRect();
  const kind = blockKind(block);
  return {
    sectionKey: section.key,
    sectionIndex,
    blockKind: kind,
    blockOrdinal: section.blocks.slice(0, blockIndex).filter((candidate) => blockKind(candidate) === kind).length,
    blockCount: section.blocks.filter((candidate) => blockKind(candidate) === kind).length,
    blockProgress: clamp((inset - rect.top) / Math.max(rect.height, 1)),
    sectionProgress: clamp((inset - start) / Math.max(last.bottom - start, 1)),
    headerProgress: sections[0].blocks[0].getBoundingClientRect().top > inset
      ? clamp(window.scrollY / Math.max(window.scrollY + sections[0].blocks[0].getBoundingClientRect().top - inset, 1))
      : null,
  };
}

export function readingPositionAtAnchor(anchor: HTMLElement): ReadingPosition | null {
  const sections = readingSections();
  const sectionIndex = sections.findIndex(section => section.blocks.includes(anchor) || section.key === `question:${anchor.dataset.questionId}`);
  if (sectionIndex < 0) return null;
  const section = sections[sectionIndex];
  const blockIndex = Math.max(0, section.blocks.indexOf(anchor));
  const kind = blockKind(section.blocks[blockIndex]);
  return {
    sectionKey: section.key, sectionIndex, blockKind: kind,
    blockOrdinal: section.blocks.slice(0, blockIndex).filter(block => blockKind(block) === kind).length,
    blockCount: section.blocks.filter(block => blockKind(block) === kind).length,
    blockProgress: 0, sectionProgress: 0, headerProgress: null,
  };
}

export function restoreReadingPosition(position: ReadingPosition): void {
  const sections = readingSections();
  if (!sections.length) return;
  if (position.headerProgress !== null) {
    const headerHeight = window.scrollY + sections[0].blocks[0].getBoundingClientRect().top - readingInset();
    window.scrollTo({ top: Math.max(0, headerHeight * position.headerProgress), behavior: "instant" });
    return;
  }
  const section = sections.find((candidate) => candidate.key === position.sectionKey)
    ?? sections[Math.min(position.sectionIndex, sections.length - 1)];
  const matchingBlocks = section.blocks.filter((candidate) => blockKind(candidate) === position.blockKind);
  // Some translations split or combine paragraphs. Preserve relative progress
  // through that block type instead of drifting toward the start of the section.
  const ordinal = matchingBlocks.length === position.blockCount ? position.blockOrdinal
    : Math.round(position.blockOrdinal / Math.max(position.blockCount - 1, 1) * (matchingBlocks.length - 1));
  const block = matchingBlocks[ordinal];
  let top: number;
  if (block) {
    const rect = block.getBoundingClientRect();
    top = rect.top + rect.height * position.blockProgress;
  } else {
    const first = section.blocks[0].getBoundingClientRect();
    const last = section.blocks[section.blocks.length - 1].getBoundingClientRect();
    top = first.top + (last.bottom - first.top) * position.sectionProgress;
  }
  window.scrollTo({ top: Math.max(0, window.scrollY + top - readingInset()), behavior: "instant" });
}
