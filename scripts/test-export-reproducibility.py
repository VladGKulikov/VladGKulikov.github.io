"""Exercise real exporters in an isolated copy, never modifying published payloads."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def snapshot(root):
    paths = list((root / 'public').rglob('*')) + list((root / 'app/generated').glob('*'))
    paths += list(root.glob('*AUDIT.json')) + list(root.glob('*BUILD.json')) + list(root.glob('LECTURE_EDITION*.json'))
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def run(root, script, *args):
    subprocess.run([sys.executable, str(root / 'scripts' / script), *args], cwd=root, check=True, stdout=subprocess.DEVNULL)


def main():
    with tempfile.TemporaryDirectory(prefix='export-check-', dir=ROOT / 'work') as folder:
        stage = Path(folder)
        for name in ('public', 'app/generated', 'scripts'):
            shutil.copytree(ROOT / name, stage / name, ignore=shutil.ignore_patterns('__pycache__'))
        for p in ROOT.iterdir():
            if p.is_file() and p.suffix in {'.json', '.md'} or p.name in {'LICENSE-CODE', 'NOTICE'}:
                shutil.copy2(p, stage / p.name)
        before = snapshot(stage)
        for script in ('export-practice.py', 'export-quizzes.py'):
            run(stage, script, '--course', 'modern-llms')
        first = snapshot(stage)
        other = [p for p in before if ('/rl-for-llm/' in p or '/information-theory-for-ml/' in p) and before[p] != first[p]]
        assert not other, ('Other courses changed', other)
        for name in ('practice', 'quizzes'):
            original = json.loads((ROOT / f'app/generated/{name}.json').read_text(encoding='utf8'))['modules']
            actual = json.loads((stage / f'app/generated/{name}.json').read_text(encoding='utf8'))['modules']
            assert [r for r in original if r['courseKey'] != 'modern-llms'] == [r for r in actual if r['courseKey'] != 'modern-llms']
        for script in ('export-practice.py', 'export-quizzes.py'):
            run(stage, script, '--course', 'modern-llms')
        assert first == snapshot(stage), 'Second complete Modern LLMs export changed files'
        # All-course mode is equally supported, and is tested only in the sandbox.
        for script in ('export-practice.py', 'export-quizzes.py'):
            run(stage, script)
        all_first = snapshot(stage)
        for script in ('export-practice.py', 'export-quizzes.py'):
            run(stage, script)
        assert all_first == snapshot(stage), 'Second all-course export changed files'
        report = {'status': 'PASS', 'modern_repeat_identical': True, 'all_courses_repeat_identical': True,
                  'unselected_course_files_unchanged': True, 'files_checked': len(all_first)}
        print(json.dumps(report))
        (ROOT / 'work/export-reproducibility.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
