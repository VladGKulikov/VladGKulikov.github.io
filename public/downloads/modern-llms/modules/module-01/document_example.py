"""One bounded reference check on fictional documents, not a semantic LLM judge."""
from pathlib import Path
import json


# BEGIN EDUCATIONAL CORE
def check_prepared_answer(answer, documents, reference):
    """Check a structured day count and an exact reference passage for one fixed question."""
    if type(answer.get("days")) is not int or answer["days"] < 0:
        raise ValueError("days must be a nonnegative integer")
    doc_id = answer.get("document_id")
    quote = answer.get("quote")
    if not isinstance(doc_id, str) or not isinstance(quote, str):
        raise ValueError("document_id and quote must be strings")
    answer_correct = answer["days"] == reference["days"]
    quote_exists = bool(quote) and doc_id in documents and quote in documents[doc_id]
    # The required passage is manually labelled for this question; it is not inferred.
    supported = (quote_exists and answer_correct
                 and doc_id == reference["document_id"]
                 and reference["required_quote"] in quote)
    return dict(answer_correct=answer_correct, quote_exists=quote_exists,
                supported=supported, accepted=answer_correct and supported)
# END EDUCATIONAL CORE


def run_checks():
    fixture = json.loads((Path(__file__).parent/'data/document_assistant.json').read_text(encoding='utf8'))
    checked = 0
    for lang, text in fixture['languages'].items():
        documents = text['documents']
        reference = dict(fixture['reference'], required_quote=documents['A'])
        answers = [dict(a, quote=documents[a['document_id']]) for a in fixture['prepared_answers']]
        results = [check_prepared_answer(a, documents, reference) for a in answers]
        assert results == fixture['expected']
        # Independent reference: compare each complete triple with the manually accepted one.
        for answer, result in zip(answers, results):
            expected = (answer['days'], answer['document_id'], answer['quote']) == (30, 'A', documents['A'])
            assert result['accepted'] == expected
            checked += 1
        for change in ({'document_id':'missing'}, {'quote':''}, {'quote':'invented'}, {'quote':documents['A'].split()[0]}):
            assert not check_prepared_answer(dict(answers[0], **change), documents, reference)['accepted']
            checked += 1
        for value in (True, -1, 30.0, None):
            try:
                check_prepared_answer(dict(answers[0], days=value), documents, reference)
            except ValueError:
                checked += 1
            else:
                raise AssertionError('invalid structured answer accepted')
    return {'status':'PASS','checks':checked,'model_calls':0, 'fixture':fixture['fixture_id']}


if __name__=='__main__':
    print(run_checks())
