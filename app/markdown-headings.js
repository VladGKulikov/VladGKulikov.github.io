/** @param {string} value */
function slugify(value) {
  return value.toLowerCase().replace(/[`*_()[\]{}]/g, "")
    .replace(/[^a-zа-яё0-9]+/gi, "-").replace(/^-|-$/g, "").slice(0, 80);
}

/**
 * Match the reader's Markdown headings and keep every generated anchor unique.
 * @param {string} markdown
 * @returns {{line: number, depth: number, label: string, id: string}[]}
 */
export function markdownHeadings(markdown) {
  const headings = [];
  const used = new Set();
  let fence = "";
  for (const [index, line] of markdown.split(/\r?\n/).entries()) {
    const marker = line.match(/^\s{0,3}(`{3,}|~{3,})(.*)$/);
    if (marker) {
      if (!fence) fence = marker[1];
      else if (marker[1][0] === fence[0] && marker[1].length >= fence.length && !marker[2].trim()) fence = "";
      continue;
    }
    if (fence) continue;
    const match = line.match(/^(#{2,4})[ \t]+(.+?)[ \t]*$/);
    if (!match) continue;
    const raw = match[2].replace(/[ \t]+#+[ \t]*$/, "");
    const base = slugify(raw) || "section";
    let id = base;
    let suffix = 2;
    while (used.has(id)) id = `${base}-${suffix++}`;
    used.add(id);
    headings.push({ line: index + 1, depth: match[1].length, label: raw.replace(/[`*_]/g, ""), id });
  }
  return headings;
}
