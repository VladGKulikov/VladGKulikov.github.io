"""Canonical notebook cells, using exactly the reviewed reference implementations."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def definitions(names):
    source = (ROOT / "reference_calculations.py").read_text(encoding="utf-8")
    nodes = {n.name: n for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    lines = source.splitlines()
    pieces = []
    for name in names:
        node = nodes[name]
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        pieces.append("\n".join(lines[start-1:node.end_lineno]))
    return "\n\n".join(pieces)


def cells(lang, text):
    ru = lang == "ru"
    result = []
    def add(md, code):
        result.extend([("markdown", md), ("code", code)])
    payload = json.loads((ROOT / "data/reference_metrics.json").read_text(encoding="utf-8"))
    payload["datasets"] = {lang: payload["datasets"][lang]}
    payload["retrieval_results"] = {lang: payload["retrieval_results"][lang]}
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    add("## Данные и границы опыта" if ru else "## Data and scope", f'''import json, math, re, hashlib
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Mapping, Sequence
import numpy as np
WORD_RE = re.compile(r"[^\\W_]+", re.UNICODE)
REFERENCE_JSON = {blob!r}
assert hashlib.sha256(REFERENCE_JSON.encode()).hexdigest() == {hashlib.sha256(blob.encode()).hexdigest()!r}
REFERENCE = json.loads(REFERENCE_JSON)
LANG = {lang!r}
DATASETS = REFERENCE["datasets"]
data = DATASETS[LANG]
ids = [row["document_id"] for row in data["documents"]]
texts = [row["text"] for row in data["documents"]]
documents = {{row["document_id"]: row for row in data["documents"]}}
queries = data["queries"]
stored = {{}}
''')
    result.insert(0, ("markdown",
        "13 авторских диагностических документов и семь запросов заморожены отдельно от лекции. Это набор для разработки, а не независимый тест: по этим запросам нельзя выбирать настройки и затем заявлять перенос качества. Языковые наборы различаются формулировками; их числа не измеряют превосходство одного языка. Регулярное выражение выделяет слова, а не токены конкретной LLM; ниже бюджет измеряется этими словесными единицами."
        if ru else
        "Thirteen authored diagnostic documents and seven development queries are frozen independently of the lecture. They are not a held-out benchmark. Tuning on these queries cannot establish generalization. The language collections use different wording, so their scores do not measure a language advantage. Our tokenizer extracts words with a regular expression; context budgets below use these word units, not any LLM's subword tokens."))
    add(text["chunk_md"], definitions(["_positive_int","_positive_finite","tokenize","Chunk","chunk_document","analytical_chunk_count","_validate_ids","_validate_judgments","stable_ranking","SearchResult"])+'''
sample = "one two three four five six seven eight nine ten"
chunks = chunk_document("sample", sample, chunk_size=4, overlap=1)
assert len(chunks) == analytical_chunk_count(10, 4, 1) == 3
assert set(i for c in chunks for i in range(c.start_token,c.end_token)) == set(range(10))
print([(c.start_token,c.end_token,c.text) for c in chunks])
original = "TS-999:  UPDATE\\nfailed — İstanbul; Straße."
quotes = chunk_document("case", original, chunk_size=4, overlap=1)
assert all(c.text == original[c.start_char:c.end_char] for c in quotes)
assert quotes[0].text == "TS-999:  UPDATE\\nfailed"
assert tokenize(quotes[-1].text) == tokenize(original)[quotes[-1].start_token:quotes[-1].end_token]
''')
    add(text["bm25_md"], definitions(["BM25Index"])+'''
bm25 = BM25Index(ids, texts)
for q in queries:
    stored[q["query_id"]] = {"bm25": bm25.search(q["text"])}
assert [r.document_id for r in stored["q0"]["bm25"]] == REFERENCE["retrieval_results"][LANG]["rankings"]["bm25"]["q0"]
print(stored["q0"]["bm25"][:3])
''')
    add(text["dense_md"], definitions(["LSAIndex"])+'''
dense = LSAIndex(ids, texts, dimension=4)
assert dense.document_vectors.shape == (13,4)
assert np.allclose(np.linalg.norm(dense.document_vectors,axis=1),1)
for q in queries:
    stored[q["query_id"]]["dense_lsa"] = dense.search(q["text"])
print(dense.document_vectors.shape)
''')
    add(text["rrf_md"].replace("линейное слияние","накопление вкладов и сортировка"), definitions(["FusedResult","reciprocal_rank_fusion"])+'''
for q in queries:
    row = stored[q["query_id"]]
    row["rrf"] = reciprocal_rank_fusion(
        {name:[r.document_id for r in row[name]] for name in ("bm25","dense_lsa")},
        depth=8)
print([asdict(r) for r in stored["q0"]["rrf"][:3]])
''')
    add(text["rerank_md"]+("\n\nСходство здесь — мера Жаккара по символьным триграммам. Это необучаемая лексическая эвристика, усредняющая максимумы по словам запроса; она не является ColBERT или кросс-энкодером. Отсечение до восьми кандидатов ограничивает достижимую полноту."
        if ru else "\n\nToken similarity is character-trigram Jaccard. This untrained lexical heuristic averages each query word's best match; it is neither ColBERT nor a cross-encoder. Truncating the candidate set to eight limits attainable recall."),
        definitions(["_character_trigrams","_token_similarity","token_interaction_score","rerank_candidates"])+'''
for q in queries:
    row = stored[q["query_id"]]
    row["reranked"] = rerank_candidates(q["text"], [r.document_id for r in row["rrf"][:8]],
                                       dict(zip(ids,texts)))
print(stored["q0"]["reranked"][:3])
''')
    add(text["metrics_md"], definitions(["reciprocal_rank","recall_at_k","ndcg_at_k","evaluate_rankings"])+'''
# No search or scoring calls inside evaluation: consume the stored records.
metrics = {}
for method in ("bm25","dense_lsa","rrf","reranked"):
    rankings = {q["query_id"]:[r.document_id for r in stored[q["query_id"]][method]]
                for q in queries}
    metrics[method] = evaluate_rankings(rankings,queries,k=5)
    assert all(np.isclose(value, REFERENCE["retrieval_results"][LANG]["metrics"][method][name])
               for name,value in metrics[method].items())
print(json.dumps(metrics,indent=2))
assert len(stored) == 7 and sum(len(row["reranked"]) for row in stored.values()) == 56
''')
    add(text["context_md"]+("\n\nСтоимость каждого фрагмента считается после добавления цитаты. Проверяется и полная сериализованная строка. Для реальной модели замените счётчик её токенизатором и дополнительно зарезервируйте инструкции, запрос, шаблон диалога и ответ."
        if ru else "\n\nEach fragment is charged after its citation is added, and the final serialized string is counted again. For an actual model, use its tokenizer and reserve instructions, the query, chat-template tokens, and output capacity."),
        definitions(["ContextItem","ContextPack","pack_context"])+'''
head = stored["q6"]["reranked"]
pack = pack_context([r.document_id for r in head],{r.document_id:r.score for r in head},
                    documents,max_tokens=42)
assert len(tokenize(pack.rendered_text)) == pack.used_tokens <= 42
expected = REFERENCE["retrieval_results"][LANG]["context_pack_q6"]
assert [r.document_id for r in pack.items] == expected["selected_ids"]
print(pack.rendered_text)
print(pack.used_tokens, pack.skipped)
''')
    add("## Сквозной запрос: от исходного текста до цитаты\n\nТеперь окна действительно поступают в индекс. Исходные 13 документов делятся на окна по 12 слов с перекрытием 3, а не только используются для отдельной проверки разбиения. Ответом служит дословная выдержка из выбранного окна. Она позволяет проверить происхождение ответа, но не демонстрирует способность LLM обобщать или связывать несколько источников. Наличие корректной ссылки ещё не доказывает, что выдержка отвечает на вопрос."
        if ru else
        "## End-to-end request: source text to cited excerpt\n\nThis time the chunks themselves enter the index. The thirteen source documents become twelve-word windows with three-word overlap. The answer is an exact excerpt from a selected window: this tests provenance, not an LLM's synthesis ability. A valid citation does not establish that the excerpt answers the question.",
        '''
all_chunks = [c for doc in data["documents"]
              for c in chunk_document(doc["document_id"],doc["text"],chunk_size=12,overlap=3)]
chunk_docs = {c.chunk_id:asdict(c) for c in all_chunks}
chunk_ids = list(chunk_docs)
chunk_texts = [chunk_docs[i]["text"] for i in chunk_ids]
lex = BM25Index(chunk_ids,chunk_texts)
vec = LSAIndex(chunk_ids,chunk_texts,dimension=4)
query = queries[6]["text"]
# Two search calls, one bounded rerank, one pack; metrics never repeat them.
branches = {"lex":[r.document_id for r in lex.search(query,top_k=8)],
            "lsa":[r.document_id for r in vec.search(query,top_k=8)]}
fused = reciprocal_rank_fusion(branches)
ranked = rerank_candidates(query,[r.document_id for r in fused[:8]],
                           dict(zip(chunk_ids,chunk_texts)))
context = pack_context([r.document_id for r in ranked],{r.document_id:r.score for r in ranked},
                       chunk_docs,max_tokens=42)
item = context.items[0]
answer = {"text":item.text, "citation":item.citation, "chunk_id":item.document_id}
record = chunk_docs[answer["chunk_id"]]
source = documents[record["source_id"]]["text"]
assert answer["text"] == source[record["start_char"]:record["end_char"]]
assert tokenize(answer["text"]) == tokenize(source)[record["start_token"]:record["end_token"]]
trace = {"search_calls":2,"rerank_calls":1,"scored_pairs":len(ranked),
         "answer":answer,"used_word_units":context.used_tokens}
assert json.loads(json.dumps(trace)) == trace
print(json.dumps(trace,ensure_ascii=False,indent=2))
''')
    add(text["accounting_md"].replace("нижняя граница","условная вычислительная оценка").replace("floor","scenario"),
        definitions(["index_storage_gib","brute_ann_scenario","cascade_latency_scenario","long_context_vs_rag"])+'''
storage = index_storage_gib(10_000_000,1024,1)
assert storage == 1_280_000_000 / 2**30
assert analytical_chunk_count(20_000_000,400,64) == 59_524
cascade = cascade_latency_scenario()
assert np.isclose(cascade["total_ms"],1658.3129827096177)
assert np.isclose(long_context_vs_rag()["ratio"],200_000/3712)
print(json.dumps(cascade,indent=2))
''')
    add(text["failure_md"], '''
def rejects(fn):
    try:
        fn()
    except (ValueError,TypeError,KeyError):
        return
    raise AssertionError("invalid input was accepted")
rejects(lambda: bm25.search("zzzzunseen"))
rejects(lambda: dense.search("zzzzunseen"))
rejects(lambda: chunk_document("x","text",chunk_size=4,overlap=4))
rejects(lambda: reciprocal_rank_fusion({"x":["a","a"]}))
rejects(lambda: ndcg_at_k(["a","a"],{"a":3},2))
rejects(lambda: ndcg_at_k(["a"],{"a":1.9},1))
rejects(lambda: evaluate_rankings({},[],k=5))
rejects(lambda: BM25Index([1,"1"],["a","b"]))
rejects(lambda: rerank_candidates("a",[1],{"1":"a"}))
rejects(lambda: pack_context([ids[0]],{ids[0]:float("nan")},documents,max_tokens=50))
rejects(lambda: pack_context([ids[0]],{ids[0]:1},documents,max_tokens=1))
assert reciprocal_rank([],{"a":1}) == 0
assert recall_at_k(["a"],{"a":1,"b":1},1) == 0.5
assert ndcg_at_k(["a","b"],{"a":3,"b":1},2) == 1
assert analytical_chunk_count(10**20+1,4,1) == (10**20+1-4+2)//3+1
print("PASS")
''')
    return result
