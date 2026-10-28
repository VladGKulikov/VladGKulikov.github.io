"""Assert every word and illustration survives the RL lecture export."""
from pathlib import Path
import importlib.util
import json
import re
import sys

SITE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('rl_canon_builder', SITE / 'scripts/refresh-course-content.py')
builder = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = builder
spec.loader.exec_module(builder)


def normalize(text):
    text = re.sub(r'<!--.*?-->', '', text, flags=re.S)
    text = re.sub(r'<a\s+(?:id|name)=[^>]*>\s*</a>', '', text)
    text = builder.normalize_reader_markup(builder.normalize_math_delimiters(text))
    text = re.sub(r'(!?\[[^\[\]\n]*\])\([^)\s]+\)', r'\1(URL)', text)
    return text.split()


root = Path(sys.argv[1]).resolve()
image_count = 0
for language in ('ru', 'en'):
    for number in range(1, 14):
        source = (root / f'modules/module-{number:02}/Module_{number}_{language.upper()}.md').read_text(encoding='utf8')
        exported = (SITE / f'public/content/rl-for-llm/{language}/module-{number:02}.md').read_text(encoding='utf8')
        assert normalize(source) == normalize(exported), (language, number, 'prose differs')
        images = lambda value: [(alt, Path(url).name) for alt, url in re.findall(r'!\[([^\]\n]*)\]\(([^)\s]+)\)', value)]
        assert images(source) == images(exported), (language, number, 'illustrations differ')
        image_count += len(images(source))
assert image_count == 220
print(json.dumps({'status': 'PASS', 'lectures': 26, 'images': image_count, 'prose': 'all words preserved'}))
