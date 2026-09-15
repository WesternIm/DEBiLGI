from threading import Lock
from types import SimpleNamespace

from debilgi.language import syntax_backend
from debilgi.language.structured_utterance import analyze_structured_utterance
from debilgi.language.syntax_backend import SlovnetParser, SyntaxEdge


def test_ud_alignment_changes_roles_without_losing_negation(monkeypatch):
    monkeypatch.setattr(syntax_backend, "syntax_edges", lambda tokens: (
        SyntaxEdge(2, 0, "obj"), SyntaxEdge(2, 1, "advmod"),
        SyntaxEdge(-1, 2, "root"), SyntaxEdge(2, 3, "nsubj"),
    ))
    result = analyze_structured_utterance("Книгу не читает мальчик")
    frame = result.clauses[0]
    assert result.tokens[frame.subject_indices[0]].lemma == "мальчик"
    assert result.tokens[frame.direct_object_indices[0]].lemma == "книга"
    assert frame.negated
    assert "syntax_backend:slovnet" in result.niche_signals
    assert any(e.head_index == 2 and e.dependent_index == 3 and e.relation == "nsubj"
               for e in result.dependencies)


def test_unaligned_root_cannot_replace_local_analysis(monkeypatch):
    monkeypatch.setattr(syntax_backend, "syntax_edges", lambda tokens: ())
    baseline = analyze_structured_utterance("Я потерял фырбик")
    monkeypatch.setattr(syntax_backend, "syntax_edges", lambda tokens: (
        SyntaxEdge(-1, 2, "root"), SyntaxEdge(2, 1, "obj"),
    ))
    assert analyze_structured_utterance("Я потерял фырбик") == baseline


def test_malformed_prediction_does_not_disable_parser_for_next_sentence():
    parser = SlovnetParser.__new__(SlovnetParser)
    parser._lock = Lock()
    parser.failed = False
    parser.model = lambda words: SimpleNamespace(tokens=(
        SimpleNamespace(id="1", head_id="1", rel="root", text=words[0]),
        SimpleNamespace(id="2", head_id="1", rel="obj", text=words[1]),
    ))
    assert parser._parse(("a", "b")) == ()
    assert not parser.failed
    parser.model = lambda words: SimpleNamespace(tokens=(
        SimpleNamespace(id="1", head_id="0", rel="root", text=words[0]),
        SimpleNamespace(id="2", head_id="1", rel="obj", text=words[1]),
    ))
    assert len(parser._parse(("a", "b"))) == 2


def test_missing_models_leave_local_grammar_available(tmp_path):
    previous = syntax_backend._parser
    try:
        assert not syntax_backend.configure_syntax_backend(tmp_path)
        assert analyze_structured_utterance("Я потерял фырбик").clauses
    finally:
        syntax_backend._parser = previous
