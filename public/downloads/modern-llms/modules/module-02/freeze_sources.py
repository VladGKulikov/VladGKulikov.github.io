"""Explicit network refresh of the small M2 source archive; never run by notebooks.

Resolve moving repository names once, fetch immutable URLs, and save original
bytes (gzip for the two large tokenizer files). Review the resulting diff before
using a new snapshot in teaching material.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import re
import urllib.request

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
ARCHIVE = DATA / 'sources_2026-10-11'


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'Modern-LLMs-course-source-check'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def field(obj, path):
    for key in path.split('.'):
        obj = obj[key]
    return obj


def main():
    old = json.loads((DATA / 'vocab_snapshot_2026-10-10.json').read_text(encoding='utf-8'))
    current = deepcopy(old)
    current['observed_at'] = '2026-10-11'
    hf_repos = {item['url'].split('/resolve/')[0].removeprefix('https://huggingface.co/')
                for item in old['configs'].values() if item['status'] == 'READ'}
    def hf_revision(repo):
        revision = json.loads(fetch('https://huggingface.co/api/models/' + repo))['sha']
        assert re.fullmatch('[0-9a-f]{40}', revision)
        return repo, revision
    with ThreadPoolExecutor(max_workers=4) as pool:
        hf = dict(pool.map(hf_revision, sorted(hf_repos)))
    gh = {}
    for repo, branch in [('openai/tiktoken', 'main'), ('google/sentencepiece', 'master'),
                         ('huggingface/transformers', 'main')]:
        revision = json.loads(fetch(f'https://api.github.com/repos/{repo}/commits/{branch}'))['sha']
        assert re.fullmatch('[0-9a-f]{40}', revision)
        gh[repo] = revision

    jobs = []
    for group in ('configs', 'tokenizer_counts', 'primary_sources'):
        for name, item in current[group].items():
            if item.get('status') == 'UNVERIFIED':
                item['observed_at'] = old['observed_at']
                continue
            url = item['url']
            revision = None
            if url.startswith('https://huggingface.co/'):
                repo, filename = url.removeprefix('https://huggingface.co/').split('/resolve/main/')
                revision = hf[repo]
                url = f'https://huggingface.co/{repo}/resolve/{revision}/{filename}'
            elif url.startswith('https://raw.githubusercontent.com/'):
                owner, repo_name, branch, filename = url.removeprefix('https://raw.githubusercontent.com/').split('/', 3)
                repo = owner + '/' + repo_name
                revision = gh[repo]
                url = f'https://raw.githubusercontent.com/{repo}/{revision}/{filename}'
            jobs.append((group, name, item, url, revision))

    ARCHIVE.mkdir(exist_ok=True)
    def download(job):
        group, name, item, url, revision = job
        raw = fetch(url)
        digest = hashlib.sha256(raw).hexdigest()
        # A new observation must still support the existing lesson's contract.
        if group == 'configs':
            obj = json.loads(raw)
            assert {key: field(obj, key) for key in item['fields']} == item['fields'], name
        elif group == 'tokenizer_counts':
            assert digest == item['sha256'], f'Changed tokenizer input: {name}; review counts separately'
        safe = re.sub('[^a-zA-Z0-9_.-]', '_', group + '__' + name)
        compressed = len(raw) > 100_000
        target = ARCHIVE / (safe + ('.gz' if compressed else '.txt'))
        target.write_bytes(gzip.compress(raw, mtime=0) if compressed else raw)
        item.update(url=url, revision=revision, sha256=digest, bytes=len(raw),
                    local_file=target.relative_to(DATA).as_posix(),
                    compression='gzip' if compressed else 'none', observed_at='2026-10-11')
        return group + '/' + name
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name in pool.map(download, jobs):
            print('PINNED', name, flush=True)
    current['protocol_source'] = current['primary_sources']['tiktoken_constructor']['url']
    current['protocol_source_evidence'] = '../validate_sources.py'
    (DATA / 'vocab_snapshot_2026-10-11.json').write_text(
        json.dumps(current, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Snapshot saved; all seven model-field contracts and tokenizer hashes unchanged.')


if __name__ == '__main__':
    main()
