import fs from 'node:fs';
import assert from 'node:assert/strict';
import { courses, buildLocation, parseUrl, progressModuleKey, quizModules, practiceModules } from '../build-static/pages-render.js';

const origin = 'https://vladkulikov.com';
const manifest = JSON.parse(fs.readFileSync('dist-pages/route-manifest.json', 'utf8'));
const oldIds = ['01','02','18','03','04','05','06','07','08','09','10','07b','11','12','13','14','15','16','17','20'];
let lectures = 0, pages = 0;
for (const course of courses) {
  assert.equal(new Set(course.modules.map(m => m.module_key)).size, course.modules.length);
  for (const courseModule of course.modules) {
    const number = Number(courseModule.position);
    const key = `module-${String(number).padStart(2, '0')}`;
    assert.equal(courseModule.module_key, key, `${course.course_key}: URL number must match reading order`);
    if (course.course_key !== 'modern-llms') continue;
    assert.equal(Number(courseModule.title.match(/^(?:Module|Модуль)\s+(\d+)\./)?.[1]), number, courseModule.title);
    assert.equal(courseModule.content_file, `modern-llms/${course.language}/${key}.md`);
    assert.equal(progressModuleKey(course.course_key, key), `module-${oldIds[number-1]}`);
    const sections = [undefined, 'tests', 'exercises', 'notebooks'];
    for (const section of sections) {
      const pathname = buildLocation(course.language, course.course_key, key, section);
      assert.equal(pathname, `${course.language === 'ru' ? '/ru' : ''}/courses/modern-llms/${key}/${section ? section+'/' : ''}`);
      const route = parseUrl(new URL(pathname, origin));
      assert.equal(route.moduleKey, key);
      assert.equal(route.section, section);
      const page = manifest.find(p => p.path === pathname);
      assert(page && page.title.startsWith(courseModule.title), pathname);
      assert(fs.existsSync(`dist-pages${pathname}index.html`));
      pages++;
    }
    const quiz = quizModules.find(e => e.courseKey === course.course_key && e.language === course.language && e.moduleKey === key);
    assert.equal(quiz.file, `modern-llms/${course.language}/${key}.json`);
    const bank = JSON.parse(fs.readFileSync(`public/quizzes/${quiz.file}`, 'utf8'));
    assert(bank.questions.every(q => q.id.startsWith(number+'.')), key);
    for (const section of ['exercises','notebooks']) {
      const entry = practiceModules.find(e => e.courseKey === course.course_key && e.language === course.language && e.moduleKey === key && e.section === section);
      assert.equal(entry.file, `modern-llms/${course.language}/${key}.${section}.json`);
    }
    lectures++;
  }
}
const sources = JSON.parse(fs.readFileSync('QUIZ_CONTENT_AUDIT.json', 'utf8')).sources;
for (const s of sources.filter(s => s.course === 'modern-llms')) {
  assert(s.source.startsWith(`modules/${s.module}/`), 'Quiz source number differs from page number');
}
assert.equal(parseUrl(new URL('/courses/modern-llms/module-07b/', origin)).notFound, true);
assert.equal(progressModuleKey('rl-for-llm','module-03'), 'module-03');
for (const name of ['sitemap.xml','llms.txt','agents.txt']) {
  assert(!fs.readFileSync(`dist-pages/${name}`, 'utf8').includes('/module-07b/'), name);
}
assert.equal(lectures, 40);
assert.equal(pages, 160);
const report = {status:'PASS', lectures, pages, unchanged_progress_topics:40};
fs.mkdirSync('work', {recursive:true});
fs.writeFileSync('work/numbering-audit.json', JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report));
