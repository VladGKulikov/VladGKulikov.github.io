import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { practiceModules, parseUrl, buildLocation, highlightPython } from '../build-static/pages-render.js';

const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const read = file => fs.readFileSync(file);
const audit = JSON.parse(read('PRACTICE_CONTENT_AUDIT.json'));
const checks = (a,b,message) => { assert.deepEqual(a,b,message); counts.assertions++; };
const counts = {pages:0, exercises:0, notebooks:0, codeCells:0, pythonAnswers:0, downloads:0, assertions:0};
const pages = new Map();
const original = new Map(audit.files.map(f => [f.url,f]));
function download(url, expected) {
  const file = path.join('public',decodeURIComponent(url));
  const bytes = read(file);
  checks(bytes.length,expected.bytes,url);
  checks(sha(bytes),expected.sha256,url);
  checks(read(path.join('dist-pages',decodeURIComponent(url))),bytes,`Built file ${url}`);
  return bytes;
}
for (const entry of practiceModules) {
  const raw=read(`public/practice/${entry.file}`).toString('utf8');
  checks(sha(raw.replaceAll('\r\n','\n')),entry.hash,entry.file);
  const page=JSON.parse(raw); pages.set(entry.file,page);
  checks([page.courseKey,page.moduleKey,page.language,page.section],[entry.courseKey,entry.moduleKey,entry.language,entry.section]);
  checks(page.items.length,entry.count,entry.file);
  checks(new Set(page.items.map(i=>i.id)).size,entry.count,'Unique task/notebook IDs');
  const url=buildLocation(entry.language,entry.courseKey,entry.moduleKey,entry.section);
  const route=parseUrl(new URL(url,'https://vladkulikov.com'));
  checks([route.language,route.courseKey,route.moduleKey,route.section],[entry.language,entry.courseKey,entry.moduleKey,entry.section],url);
  const html=read(path.join('dist-pages',url,'index.html')).toString('utf8');
  checks((html.match(/<details[^>]*data-disclosure=/g)??[]).length,entry.count,'Native expandable answers/previews');
  assert(!/<details[^>]*data-disclosure=[^>]*\sopen(?:[ =>])/.test(html),'Answers initially hidden without saved progress');
  for(const item of page.items) {
    assert(item.title && (entry.section==='notebooks' || item.prompt),`${entry.file}:${item.id} empty item`);
    if(entry.section==='exercises') {
      counts.exercises++;
      if(!item.answer) {
        checks([entry.courseKey,entry.moduleKey,item.answerKind],['modern-llms','module-13','guidance'],'Only authored project guidance may lack an answer');
        assert(['exercise-06','exercise-07'].includes(item.id),'Known project guidance IDs');
      }
      for(const file of item.codeFiles??[]) {
        counts.pythonAnswers++;
        checks(entry.courseKey,'information-theory-for-ml','Python is an exercise answer');
        checks(item.id,`exercise-${['module-14','module-15'].includes(entry.moduleKey)?'07':'06'}`,'Correct Python exercise association');
        assert(/Python/i.test(item.title));
        const bytes=download(file.download,file);
        checks(bytes.toString('utf8').replace(/^\uFEFF/,''),file.source,'Code answer is the original source');
        checks(highlightPython(file.source).map(p=>p.text).join(''),file.source,'Highlighting preserves every character');
      }
    } else {
      counts.notebooks++;
      const notebook=JSON.parse(download(item.download,item));
      assert(original.has(item.download),'Notebook provenance');
      checks(notebook.cells.length,item.cells.length,'All original notebook cells');
      checks(new Set(item.cells.map(c=>c.id)).size,item.cells.length,'Unique semantic cell IDs');
      notebook.cells.forEach((cell,index)=>{
        checks(cell.cell_type,item.cells[index].type,'Cell type');
        checks(cell.outputs??[],[],'Current source has no saved output');
        if(cell.cell_type==='code') {
          counts.codeCells++;
          const source=Array.isArray(cell.source)?cell.source.join(''):cell.source;
          checks(item.cells[index].source,source,'Original code is unchanged and inert');
          checks(highlightPython(source).map(p=>p.text).join(''),source,'Highlighting preserves original code');
        }
      });
    }
  }
  checks(page.bundle,audit.bundles[entry.courseKey],'Archive metadata');
  counts.pages++;
}
for(const entry of practiceModules.filter(e=>e.language==='ru')) {
  const other=practiceModules.find(e=>e.language==='en'&&e.courseKey===entry.courseKey&&e.moduleKey===entry.moduleKey&&e.section===entry.section);
  assert(other,'Missing translation'); checks(other.revision,entry.revision,'Shared disclosure key');
  const ru=pages.get(entry.file), en=pages.get(other.file);
  checks(ru.items.map(i=>i.id),en.items.map(i=>i.id),'Translation task/notebook alignment');
  ru.items.forEach((item,index)=>{
    const translated=en.items[index];
    if(entry.section==='notebooks') checks(item.cells.map(c=>c.id).sort(),translated.cells.map(c=>c.id).sort(),'Translation cell bookmarks');
    else {
      checks([item.answerKind,Boolean(item.answer)],[translated.answerKind,Boolean(translated.answer)],'Translation answer completeness');
      checks(item.codeFiles??[],translated.codeFiles??[],'Shared Python answer files');
    }
  });
}
for(const file of audit.files) {download(file.url,file);counts.downloads++;}
for(const bundle of Object.values(audit.bundles)) download(bundle.url,bundle);
for(const language of ['en','ru']) for(const [module,section] of [['module-01','exercises'],['module-16','exercises'],['module-07','notebooks']]) {
  checks(parseUrl(new URL(buildLocation(language,'information-theory-for-ml',module,section),'https://vladkulikov.com')).notFound,true,'Do not invent absent materials');
}
checks([counts.pages,counts.exercises,counts.notebooks,counts.pythonAnswers],[160,938,110,18],'Complete bilingual publication');
checks(Object.fromEntries(['modern-llms','rl-for-llm','information-theory-for-ml'].map(course=>[course,[...pages.values()].filter(p=>p.language==='en'&&p.courseKey===course&&p.section==='exercises').reduce((n,p)=>n+p.items.length,0)])),{'modern-llms':116,'rl-for-llm':263,'information-theory-for-ml':90});
const report={status:'PASS',counts};
fs.mkdirSync('work',{recursive:true});fs.writeFileSync('work/practice-audit.json',JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));
