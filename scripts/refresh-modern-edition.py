"""Refresh only Modern LLMs lectures; preserve quizzes, practice and other courses."""
from pathlib import Path
import hashlib, importlib.util, json, subprocess, sys
from export_io import write_text

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('course_content_builder',ROOT/'scripts/refresh-course-content.py')
builder=importlib.util.module_from_spec(spec);sys.modules[spec.name]=builder;spec.loader.exec_module(builder)
source=Path(sys.argv[1]).resolve()
context=builder.BuildContext(project_root=ROOT,content_root=ROOT/'public/content',assets_root=ROOT/'public/assets',source_roots={'modern-llms':source},missing_assets=[],copied_assets={},removed_notebook_blocks=[],omitted_sections=[])
documents=[]
for language in ('ru','en'):documents.extend(builder.build_modern({},language,context))
assert len(documents)==40 and not context.missing_assets
assert not context.omitted_sections and not context.removed_notebook_blocks
catalog_path=ROOT/'app/generated/content.json'
catalog=json.loads(catalog_path.read_text(encoding='utf8'))
fresh_courses=builder.course_catalog(documents+[d for d in catalog['documents'] if d['course_key']!='modern-llms'])
for i,course in enumerate(catalog['courses']):
    if course['course_key']=='modern-llms':catalog['courses'][i]=next(c for c in fresh_courses if c['language']==course['language'] and c['course_key']=='modern-llms')
fresh={d['id']:{k:v for k,v in d.items() if k not in {'markdown','source_file'}} for d in documents}
catalog['documents']=[fresh.get(d['id'],d) for d in catalog['documents']]
write_text(catalog_path,json.dumps(catalog,ensure_ascii=False,indent=2)+'\n')
report={'date':'2026-10-25','status':'PASS','lectures':len(documents),'source_files':{d['id']:{'path':d['source_file'],'sha256':hashlib.sha256((source/d['source_file']).read_bytes()).hexdigest()} for d in documents},'images':sum(d['image_count'] for d in documents),'practice':'unchanged'}
write_text(ROOT/'LECTURE_EDITION_2026-10-25.json',json.dumps(report,ensure_ascii=False,indent=2)+'\n')
build_path=ROOT/'CONTENT_BUILD.json'
build=json.loads(build_path.read_text(encoding='utf8'))
build['content_updated']='2026-10-25'
build['source_files']={k:v for k,v in build['source_files'].items() if not k.startswith('modern-llms:')}
build['source_files'].update(report['source_files'])
build['courses']=catalog['courses']
build['omitted_sections']=[row for row in build['omitted_sections'] if row['course_key']!='modern-llms']
build['source_repositories']['modern-llms']={
    'commit':subprocess.check_output(['git','-c','safe.directory=*','rev-parse','HEAD'],cwd=source,text=True).strip(),
    'working_tree_changes':bool(subprocess.check_output(['git','-c','safe.directory=*','diff','--name-only','--','modules','course_manifest.json'],cwd=source,text=True).strip())}
build['latest_refresh']={'scope':'Modern LLMs only','lecture_edition':'2026-10-25','inventory':'LECTURE_EDITION_2026-10-25.json','practice':'unchanged','other_courses':'unchanged'}
write_text(build_path,json.dumps(build,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='source_files'}))
