"""Regenerate the M1 classroom code, notebooks, diagrams, and HTML deterministically."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]/'tools'))
from learning_paths import minimal_source
from build_notebook_passports_shared import write_notebook
from build_m1_svgs import build
from build_m1_plots import build as build_plots
from project_demo_m01 import build_m1_project
from md2html import convert, scan


def main():
    for lang in ('ru','en'):
        path=ROOT/lang/f'module_01_lecture_{lang.upper()}.md'
        text=path.read_text(encoding='utf8')
        start=text.index('```python',text.index('## 17. '))
        end=text.index('```',start+3)+3
        code=minimal_source(lang)+'\nconfig = json.loads(EXAMPLE_CONFIG_JSON)\nprint(passport(config))\n'
        text=text[:start]+'```python\n'+code+'```'+text[end:]
        path.write_text(text,encoding='utf8')
        write_notebook(lang)
        build_m1_project(lang)
        build(lang)
        build_plots(lang)
        for source in (path,ROOT/lang/f'module_01_tests_{lang.upper()}.md'):
            html=convert(source)
            result=scan(source,html)
            assert result['OK'], result
            print(result)


if __name__=='__main__':main()
