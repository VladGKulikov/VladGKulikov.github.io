import equations from './generated/module-20-equations.json';

type MathNode = { tagName?: string; value?: string; children?: MathNode[] };

function annotation(node: MathNode): string | undefined {
  if (node.tagName === 'annotation') {
    return node.children?.map(child => child.value ?? '').join('');
  }
  for (const child of node.children ?? []) {
    const tex = annotation(child);
    if (tex !== undefined) return tex;
  }
}

// Keep the native MathML layout of the standalone edition, including its
// aligned rows and operator limits. TeX remains the key and is not rewritten.
export function module20Equation(node: MathNode | undefined, language: 'ru' | 'en'): string {
  const tex = node && annotation(node);
  const mathml = tex && (equations[language] as Record<string, string>)[tex];
  if (!mathml) throw new Error(`Missing Module 20 MathML equation (${language}): ${tex}`);
  return mathml;
}
