from debilgi.cognition.models import ActionType
from debilgi.language.grammar import SoftGrammar, role
from debilgi.language.memory_based import MemoryPhraseGenerator
from debilgi.language.planning import ReplyContext, StructuredThought, utterance_plan
from debilgi.language.tokenizer import tokenize
from debilgi.personality.expression import PersonalityExpression


def test_soft_grammar_learns_roles_and_prefers_complete_sequences() -> None:
    grammar = SoftGrammar.learn(
        (
            ("я", "смотрит", "на", "дракона", "."),
            ("он", "смотрит", "на", "город", "."),
            ("она", "видит", "странные", "огни", "."),
        )
    )

    assert role("он") == "pronoun"
    assert role("смотрит") == "verb"
    assert role("на") == "preposition"
    assert role("ответ") == "content"
    assert role("отвечающую") == "participle"
    assert grammar.score(("он", "смотрит", "на", "город", ".")) > grammar.score(
        ("на", "и", "потому")
    )


def test_generator_can_build_a_longer_multi_clause_response() -> None:
    personality = PersonalityExpression(
        genome_signature="test",
        chaos=0.35,
        curiosity=0.70,
        impulsivity=0.40,
        expressiveness=0.80,
        memory_attachment=0.80,
        novelty_drive=0.60,
        coherence_drive=0.85,
        target_tokens=36,
        candidate_count=64,
    )
    context = ReplyContext(
        prompt="why dragon watches the city",
        memory_messages=(
            "dragon watches the city and notices strange lights because people make noise "
            "at night . then it rises higher and tries to understand what happens in the square .",
            "people think the dragon is dangerous but it simply watches the city and collects "
            "new knowledge about local behavior .",
            "if lights appear again the dragon remembers the place and compares the event with "
            "the previous night .",
        ),
        seed="long-grammar",
        personality=personality,
    )
    plan = utterance_plan(
        StructuredThought(ActionType.SHORT_REPLY, None, "explain", 0.8, ()),
        context=context,
    )

    assert plan is not None
    response = MemoryPhraseGenerator().generate(plan)
    response_tokens = tokenize(response)
    assert 20 <= len(response_tokens) <= 72
    assert sum(token in {".", "!", "?"} for token in response_tokens) >= 1
