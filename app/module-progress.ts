// Browser progress belongs to a topic. URL renumbering must not attach an
// existing learner's answers or disclosures to a different topic.
const modernProgressIds: Record<string, string> = {
  'module-03': 'module-18', 'module-04': 'module-03',
  'module-05': 'module-04', 'module-06': 'module-05',
  'module-07': 'module-06', 'module-08': 'module-07',
  'module-09': 'module-08', 'module-10': 'module-09',
  'module-11': 'module-10', 'module-12': 'module-07b',
  'module-13': 'module-11', 'module-14': 'module-12',
  'module-15': 'module-13', 'module-16': 'module-14',
  'module-17': 'module-15', 'module-18': 'module-16',
  'module-19': 'module-17',
};

export function progressModuleKey(course: string, module: string): string {
  return course === 'modern-llms' ? modernProgressIds[module] ?? module : module;
}
