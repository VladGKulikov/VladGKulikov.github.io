#!/usr/bin/env python3
"""Publish existing exercises, worked answers and notebooks, without executing code."""
import argparse, hashlib, json, re, runpy, shutil, sys, zipfile
from collections import Counter
from pathlib import Path
from urllib.parse import quote, unquote

ROOT=Path(__file__).resolve().parents[1]
TOOLS=runpy.run_path(str(ROOT/'scripts/refresh-course-content.py'))
DEFAULTS=TOOLS['DEFAULT_ROOTS']
START=re.compile(r'^(?:\*\*(?P<strong>(?:(?:У|E)\d+|(?:Exercise|Упражнение)\s+\d+)[^*\n]*)\*\*[ \t]*(?P<tail>[^\n]*)|#{3,4}\s+(?P<heading>(?:Exercise|Упражнение)\s+\d+[^\n]*))$',re.M)
SOLUTION=re.compile(r'(?m)^(?:\*{1,2}(?:Решение|Solution|Ответ|Answer|Ожидаемый результат|Expected result|Критерий готовности|Completion criterion|Readiness criterion|Граница вывода|Scope of conclusion|Boundary of the result)[^*\n]*\*{1,2})',re.I)
BONUS=re.compile(r'^(?:#{3,4}\s+(?P<heading>(?:Бонус|Bonus|Инженерный бонус|Engineering bonus)[^\n]*)|\*\*(?P<strong>(?:Бонус|Bonus)\.[^*\n]*)\*\*[ \t]*(?P<tail>[^\n]*))$',re.M|re.I)
files={}; sources=[]

def sha(data): return hashlib.sha256(data).hexdigest()
def text_hash(text): return sha(text.replace('\r\n','\n').encode())
def read(path):
    raw=path.read_bytes(); sources.append({'source':str(path),'sha256':sha(raw)})
    return raw.decode('utf-8-sig')
def publish_download_bytes(course,relative,raw):
    """Adapt Capstone file instructions to the archive published by this site."""
    if course!='modern-llms' or not re.fullmatch(r'capstone/notebooks/(?:ru|en)/0[1-4]_[^/]+\.ipynb',relative):
        return raw
    data=json.loads(raw.decode('utf-8-sig'));language=relative.split('/')[2]
    archive='https://vladkulikov.com/downloads/modern-llms/course-materials.zip'
    note=(
        f'> **Файлы для этого ноутбука.** Скачайте [course-materials.zip]({archive}) на сайте курса и распакуйте архив целиком, сохранив папку `capstone`. В Colab можно загрузить этот же ZIP, когда ноутбук предложит выбрать файлы.\n'
        if language=='ru' else
        f'> **Files for this notebook.** Download [course-materials.zip]({archive}) from the course website and extract the complete archive, preserving the `capstone` folder. In Colab, upload the same ZIP when the notebook prompts for files.\n'
    )
    for cell in data['cells']:
        source=''.join(cell.get('source',[]))
        if cell['cell_type']=='code':
            if 'search_root / "capstone" / "minillm"' not in source and 'def _find_capstone_root(' in source:
                source=source.replace(
                    '    for child in sorted(search_root.glob("modern-llms-capstone-*")):',
                    f'    for package_root in (search_root / "capstone" / "minillm" / "{language}", search_root / "minillm" / "{language}"):\n'
                    '        found = _activate_capstone_root(package_root)\n'
                    '        if found:\n'
                    '            return found\n'
                    '    for child in sorted(search_root.glob("modern-llms-capstone-*")):',
                )
            if '_find_capstone_root(destination)' not in source:
                prefix='архив проекта: ' if language=='ru' else 'project archive: '
                source=source.replace(
                    '        _safe_extract(archive_path, destination)\n',
                    '        _safe_extract(archive_path, destination)\n'
                    '        root = _find_capstone_root(destination)\n'
                    '        if root:\n'
                    f'            return {json.dumps(prefix,ensure_ascii=False)} + str(archive_path)\n',
                )
            if 'glob("course-materials.zip")' not in source:
                source=source.replace('sorted(Path.cwd().glob(ARCHIVE_GLOB))', 'sorted(Path.cwd().glob(ARCHIVE_GLOB)) + sorted(Path.cwd().glob("course-materials.zip"))')
            source=source.replace('Загрузите ZIP-пакет Capstone из урока Stepik.', 'Загрузите course-materials.zip со страницы ноутбуков курса.')
            source=source.replace('Скачайте языковой ZIP-пакет Capstone из урока Stepik', 'Скачайте course-materials.zip со страницы ноутбуков курса')
            source=source.replace('Upload the Capstone ZIP package from the Stepik lesson.', 'Upload course-materials.zip from the course notebook page.')
            source=source.replace('Download the language-specific Capstone ZIP package from the Stepik lesson', 'Download course-materials.zip from the course notebook page')
            cell['source']=source.splitlines(keepends=True)
    first=next(cell for cell in data['cells'] if cell['cell_type']=='markdown')
    source=''.join(first.get('source',[]));heading,separator,rest=source.partition('\n')
    if '**Файлы для этого ноутбука.**' not in source and '**Files for this notebook.**' not in source:
        first['source']=(heading+'\n\n'+note+'\n'+rest.lstrip('\n')).splitlines(keepends=True)
    newline='\r\n' if b'\r\n' in raw else '\n'
    return (json.dumps(data,ensure_ascii=False,indent=1)+'\n').replace('\n',newline).encode('utf8')

def copy_file(course,root,path):
    path=path.resolve(); relative=path.relative_to(root.resolve()).as_posix()
    assert path.is_file(),path
    assert path.suffix.lower() in {'.ipynb','.py','.json','.csv','.txt','.yaml','.yml','.svg','.png','.jpg','.jpeg','.webp','.npz','.npy','.md'},path
    target=ROOT/'public/downloads'/course/relative
    original=path.read_bytes();raw=publish_download_bytes(course,relative,original)
    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    entry={'source':relative,'url':'/downloads/'+course+'/'+quote(relative),'sha256':sha(raw),'bytes':len(raw)}
    if raw!=original:entry.update(source_sha256=sha(original),publication_edits=['capstone-website-downloads-v1'])
    files[(course,relative)]=entry
    return entry

def clean(source,course,root,path):
    # Preserve fenced code byte-for-byte; only normalize authored Markdown.
    parts=re.split(r'(^```[^\n]*\n.*?^```[^\n]*$)',source,flags=re.M|re.S)
    for i,s in enumerate(parts):
        if i%2:continue
        s=re.sub(r'<!--.*?-->','',s,flags=re.S)
        s=re.sub(r'<a\s+id="[^"]+"\s*>\s*</a>','',s,flags=re.I)
        s=re.sub(r'</?(?:strong|b)>','**',s,flags=re.I);s=re.sub(r'</?(?:em|i)>','*',s,flags=re.I)
        s=re.sub(r'<br\s*/?>','\n',s,flags=re.I)
        s=re.sub(r'(?m)^(\s*)\\[\[\]]\s*$',r'\1$$',s)
        # A reviewed source typo closes this Wilson interval with a backtick.
        s=s.replace('$[0,0.1611]`','$[0,0.1611]$')
        # Separate resumed numbered instructions from their preceding prose.
        s=re.sub(r'(?m)^(\d{1,2}\.\s+)',r'\n\1',s)
        def link(m):
            url=m[2]
            if url.startswith('https://vladkulikov.com/downloads/'):
                return f'{m[1]}({url.removeprefix("https://vladkulikov.com")})'
            if re.match(r'^(?:https?:|mailto:|#|/|data:)',url):return m[0]
            value=unquote(url.split('#')[0]); resolved=(path.parent/value).resolve()
            entry=copy_file(course,root,resolved)
            return f'{m[1]}({entry["url"]})'
        s=re.sub(r'(!?\[[^\]\n]*\])\(([^)\s]+)\)',link,s)
        parts[i]=s
    result=TOOLS['normalize_reader_markup'](TOOLS['normalize_math_delimiters'](''.join(parts)))
    # Closing display delimiters cannot share a line with following prose.
    result=re.sub(r'(?m)^(\$\$)[ \t]+(\S.*)$',r'\1\n\n\2',result)
    return result.strip()

def exercise_section(text):
    matches=list(re.finditer(r'^##\s+.*?(?:Упражнения|Exercises).*$',text,re.M|re.I))
    assert matches,'No exercise section'
    match=matches[-1];tail=text[match.end():]
    return re.split(r'^##\s+',tail,maxsplit=1,flags=re.M)[0].strip()

def split_exercises(section,bonus=False):
    matches=list(START.finditer(section))
    if bonus: matches+=list(BONUS.finditer(section))
    matches.sort(key=lambda m:m.start())
    result=[]
    for i,m in enumerate(matches):
        groups=m.groupdict()
        title=groups.get('strong') or groups.get('heading') or (m[0].split('**')[1] if m[0].startswith('**') else m[0].lstrip('# '))
        tail=groups.get('tail') or ''
        body=tail+'\n'+section[m.end():matches[i+1].start() if i+1<len(matches) else len(section)]
        # The last exercise is followed by the lecture's concluding prose.
        body=re.split(r'^---\s*$',body,maxsplit=1,flags=re.M)[0].strip()
        number=re.search(r'\d+',title)
        key=f'exercise-{int(number[0]):02d}' if number and START.fullmatch(m[0]) else 'bonus'
        result.append({'id':key,'title':title.strip().strip(' .*'),'body':body})
    assert result and len(set(r['id'] for r in result))==len(result),[r['title'] for r in result]
    return section[:matches[0].start()].strip(),result

def modern_exercises(path,course,root):
    intro,records=split_exercises(exercise_section(read(path)))
    items=[]
    for r in records:
        mark=SOLUTION.search(r['body'])
        project=bool(re.search(r'\[(?:проект|project)\]',r['title'],re.I))
        if mark:
            prompt,answer=r['body'][:mark.start()],r['body'][mark.start():]
        else:
            demo=re.search(r'^>.*?(?:Демо|Demo|demo)',r['body'],re.M)
            assert project,(path,r['id'],'No authored answer')
            prompt,answer=(r['body'][:demo.start()],r['body'][demo.start():]) if demo else (r['body'],'')
        items.append({'id':r['id'],'title':r['title'],'prompt':clean(prompt,course,root,path),'answer':clean(answer,course,root,path),
            'answerKind':'guidance' if project else 'solution'})
    return clean(intro,course,root,path),items

def information_exercises(path,solution_path,course,root):
    intro,records=split_exercises(exercise_section(read(path)),bonus=True)
    solution=read(solution_path)
    pattern=re.compile(r'^#{1,3}\s+((?:Упражнение|Exercise)\s+(\d+)[^\n]*|(?:Бонус|Bonus)\.[^\n]*)$',re.M|re.I)
    matches=list(pattern.finditer(solution));answers={}
    for i,m in enumerate(matches):
        key=f'exercise-{int(m[2]):02d}' if m[2] else 'bonus'
        answers[key]=solution[m.end():matches[i+1].start() if i+1<len(matches) else len(solution)].strip()
    assert set(answers)=={r['id'] for r in records},(path,set(answers),[(r['id'],r['title']) for r in records])
    return clean(intro,course,root,path),[{'id':r['id'],'title':r['title'],'prompt':clean(r['body'],course,root,path),
        'answer':clean(answers[r['id']],course,root,solution_path),'answerKind':'solution'} for r in records]

def notebook(path,course,root,key,language,kind):
    read(path);resource=copy_file(course,root,path)
    data=json.loads((ROOT/'public'/unquote(resource['url']).lstrip('/')).read_text(encoding='utf-8-sig'))
    cells=[];counts=Counter()
    for cell in data['cells']:
        cell_type=cell['cell_type'];source=''.join(cell.get('source',[]));counts[cell_type]+=1
        cell_key=f'{cell_type}-{counts[cell_type]:03d}'
        if cell_type=='markdown':
            heading=source.splitlines()[0] if source.splitlines() else ''
            # The RoPE translations move the final bridge around the exercise
            # cell. Give those two cells semantic keys; keep all others ordinal.
            if path.name=='module_04_notebook_rope.ipynb':
                bridge=re.search(r'B\d{2}',heading)
                if bridge:cell_key='bridge-'+bridge[0].lower()
                elif re.search(r'^#+\s+(?:Практические задания|Exercises)\s*$',heading,re.I):cell_key='exercises'
            source=clean(source,course,root,path)
        cells.append({'id':cell_key,'type':cell_type,'source':source})
        # Current canonical notebooks contain no saved output. Fail on a future
        # rich-output release so it receives an explicit inert preview review.
        assert not cell.get('outputs'),(path,'Saved outputs require preview review')
    assert len({c['id'] for c in cells})==len(cells),path
    title=re.search(r'^#\s+(.+)$',next((''.join(c['source']) for c in data['cells'] if c['cell_type']=='markdown'),''),re.M)
    return {'id':key,'kind':kind,'title':title[1] if title else path.stem,'format':'ipynb','filename':path.name,
        'download':resource['url'],'bytes':resource['bytes'],'sha256':resource['sha256'],'cells':cells}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',action='append',default=[])
    args=parser.parse_args();roots=dict(DEFAULTS)
    for value in args.source_root:
        key,value=value.split('=',1);roots[key]=Path(value)
    sys.path.insert(0,str(roots['rl-for-llm']/'tools'))
    from assessment_source import load_canonical_tasks
    catalog=json.loads((ROOT/'app/generated/content.json').read_text(encoding='utf8'))
    modern=json.loads((roots['modern-llms']/'course_manifest.json').read_text(encoding='utf8'))['modules']
    modern_ids={f'module-{m["id"]}':m['id'] for m in modern}
    pages=[]
    for course in catalog['courses']:
        key,language=course['course_key'],course['language'];root=roots[key]
        for module in course['modules']:
            mk=module['module_key'];n=int(module['position']);items=[];intro='';books=[]
            if key=='modern-llms':
                current=modern_ids[mk];folder=root/f'modules/module-{current}/{language}'
                record=next(m for m in modern if m['id']==current)
                source=root/record['practice_sources'][language] if record.get('practice_sources') else next(folder.glob(f'*lecture_{language.upper()}.md'))
                intro,items=modern_exercises(source,key,root)
                nb=list(folder.glob('*.ipynb'));assert len(nb)==1,(folder,nb)
                books.append(notebook(nb[0],key,root,'main',language,'main'))
                demo=root/f'project_demos/{language}/module_{current}_project_demo.ipynb'
                if demo.exists():books.append(notebook(demo,key,root,'project-demo',language,'project'))
                if n==20:
                    for f in sorted((root/f'capstone/notebooks/{language}').glob('*.ipynb')):
                        books.append(notebook(f,key,root,'capstone-'+f.stem.split('_')[0],language,'capstone'))
            elif key=='rl-for-llm':
                source=root/f'modules/module-{n:02d}/Module_{n}_Assessment_{language.upper()}.md';read(source)
                items=[{'id':r['id'],'title':r['title'],'prompt':clean(r['body'],key,root,source),'answer':clean(r['solution'],key,root,source),'answerKind':'solution'} for r in load_canonical_tasks(root,n,language.upper())]
                nb=list((root/'notebooks').glob(f'module-{n:02d}-*.{language}.ipynb'));assert len(nb)==1,nb
                books.append(notebook(nb[0],key,root,'main',language,'main'))
            elif n not in (1,16):
                source=TOOLS['information_theory_source'](root,n,language)
                solution=root/f'Module_{n:02d}/Module_{n}__Exercise_solutions{"_RU" if language=="ru" else ""}.md'
                intro,items=information_exercises(source,solution,key,root)
                programs=list((root/f'Module_{n:02d}').glob(f'M{n}_*.py'))
                if programs:
                    assert len(programs)==1,programs
                    targets=[item for item in items if re.search(r'\bPython\b',item['title'],re.I)]
                    assert len(targets)==1,(n,[i['title'] for i in targets])
                    p=programs[0];entry=copy_file(key,root,p)
                    targets[0]['codeFiles']=[{'id':'python-solution','filename':p.name,'download':entry['url'],'bytes':entry['bytes'],'sha256':entry['sha256'],'source':read(p),'language':'python'}]
            for section,payload in [('exercises',{'introduction':intro,'items':items}),('notebooks',{'items':books})]:
                if payload['items']:
                    pages.append({'courseKey':key,'moduleKey':mk,'language':language,'section':section,**payload})
    # Include original support files in an offline archive, preserving source
    # hierarchy. Exclude authoring generators, caches, HTML and internal audits.
    for course,root in roots.items():
        if course=='information-theory-for-ml':continue
        bases=[root/'modules',root/'notebooks',root/'configs',root/'capstones'] if course=='rl-for-llm' else [root/'modules',root/'project_demos',root/'capstone/minillm',root/'capstone/configs',root/'capstone/data',root/'bridges']
        for base in bases:
            if not base.exists():continue
            for p in sorted(base.rglob('*')):
                if not p.is_file() or any(part.startswith('.') or part in {'__pycache__','node_modules','tmp','audits','build','checkpoints','runs'} for part in p.relative_to(base).parts):continue
                if p.suffix.lower() not in {'.py','.json','.csv','.txt','.yaml','.yml'}:continue
                if p.name.startswith(('build_','generate_','render_','audit_','validate_','repair_')):continue
                copy_file(course,root,p)
    bundles={}
    for course in roots:
        destination=ROOT/f'public/downloads/{course}/course-materials.zip';destination.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            for (key,relative),entry in sorted(files.items()):
                if key!=course:continue
                info=zipfile.ZipInfo(relative,date_time=(2026,10,16,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
                archive.writestr(info,(ROOT/'public'/unquote(entry['url']).lstrip('/')).read_bytes())
            for name in ['LICENSE.md','LICENSE-CONTENT.md','LICENSE-CODE','NOTICE']:
                info=zipfile.ZipInfo(name,date_time=(2026,10,16,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;archive.writestr(info,(ROOT/name).read_bytes())
        raw=destination.read_bytes();bundles[course]={'url':f'/downloads/{course}/course-materials.zip','bytes':len(raw),'sha256':sha(raw)}
    entries=[]
    for page in pages:
        counterpart=next(p for p in pages if p['courseKey']==page['courseKey'] and p['moduleKey']==page['moduleKey'] and p['section']==page['section'] and p['language']!=page['language'])
        assert [i['id'] for i in page['items']]==[i['id'] for i in counterpart['items']],(page['courseKey'],page['moduleKey'],page['section'],'Translation item mismatch')
        if page['section']=='exercises':
            assert all(i['title'] and i['prompt'] for i in page['items']),(page['moduleKey'],'Empty exercise')
            assert [(i['answerKind'],bool(i['answer'])) for i in page['items']]==[(i['answerKind'],bool(i['answer'])) for i in counterpart['items']],(page['moduleKey'],'Translation answer completeness')
        if page['section']=='notebooks':
            for a,b in zip(page['items'],counterpart['items']):
                assert {c['id'] for c in a['cells']}=={c['id'] for c in b['cells']},(page['moduleKey'],a['id'],'Notebook cell alignment')
        page['bundle']=bundles[page['courseKey']]
        file=f'{page["courseKey"]}/{page["language"]}/{page["moduleKey"]}.{page["section"]}.json';raw=json.dumps(page,ensure_ascii=False,indent=2)+'\n'
        destination=ROOT/'public/practice'/file;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(raw,encoding='utf8')
        revision=text_hash(json.dumps([i['id'] for i in page['items']]))[:16]
        entries.append({k:page[k] for k in ['courseKey','moduleKey','language','section']}|{'file':file,'hash':text_hash(raw),'revision':revision,'count':len(page['items'])})
    (ROOT/'app/generated/practice.json').write_text(json.dumps({'modules':entries},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    for entry in sources:
        path=Path(entry['source']);course=next(key for key,root in roots.items() if path.is_relative_to(root));entry['course']=course;entry['source']=path.relative_to(roots[course]).as_posix()
    report={'status':'PASS','sources':sources,'files':[dict(course=course,**entry) for (course,_),entry in sorted(files.items())],'bundles':bundles,
        'counts':{'localizedPages':len(entries),'exercisesPerLanguage':{key:sum(len(p['items']) for p in pages if p['language']=='en' and p['section']=='exercises' and p['courseKey']==key) for key in roots},'notebooksPerLanguage':sum(len(p['items']) for p in pages if p['language']=='en' and p['section']=='notebooks'),'itPythonSolutions':9}}
    (ROOT/'PRACTICE_CONTENT_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report['counts'],ensure_ascii=False))

if __name__=='__main__':main()
