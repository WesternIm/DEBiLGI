"""Stage 4.24 («Этап A грамматиста»): the plan as a tree, no network.

Every example uses lemmas that were not in the live dialogue or in earlier tests (врач, пилот,
кот, дерево, художник …): the mechanism (head link, government of the embedded verb, likeness)
must generalise, not remember words.
"""

import json
from dataclasses import replace

import pytest

from debilgi.language.communication import MessagePlan, ReaderGoal
from debilgi.language.dialogue_obligations import thought_from_data
from debilgi.language.plan_tree import thought_to_tree
from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    DependencyPattern,
    SemanticDependency,
    identity_graph,
    productive_pattern_curriculum,
    realize_thought,
)
from debilgi.language.semantic_graph import (
    complete_nested_complement,
    parse_proposition,
    proposition_key,
)
from debilgi.language.structured_utterance import analyze_structured_utterance

SELF = ConceptArgument(
    ConceptLexeme("я", "я", "pronoun", (("number", "singular"), ("person", "first"))),
    grammatical_case="nominative",
)


def verb(lemma, *features):
    return ConceptLexeme(lemma, lemma, "verb", (("tense", "present"), *features))


def noun(lemma, case=None, **kwargs):
    return ConceptArgument(
        ConceptLexeme(lemma, lemma, "noun", (("number", "singular"),)),
        grammatical_case=case, **kwargs)


def modal_thought(nested, *attached, mood=None, head="action_complement"):
    """«я хочу <nested> <attached…>»: attached arguments hang on the embedded verb."""
    features = (("mood", mood),) if mood else ()
    return ClauseThought(
        verb("хотеть", *features),
        (SemanticDependency("subject", SELF),
         SemanticDependency("action_complement", ConceptArgument(
             ConceptLexeme(nested, nested, "verb"))),
         *(SemanticDependency(role, argument, head=head) for role, argument in attached)),
        predicate_class="modal", confidence=0.8)


def say(thought, limit=1):
    return [s.text for s in realize_thought(thought, productive_pattern_curriculum(), limit=limit)]


# ------------------------------------------------------------------ head: the tree link

def test_default_head_is_the_predicate_and_identity_is_byte_for_byte_the_old_one():
    flat = ClauseThought(verb("читать"), (
        SemanticDependency("subject", SELF), SemanticDependency("direct_object", noun("книга"))))
    assert all(d.head is None for d in flat.dependencies)
    graph = identity_graph(flat)
    assert all("head" not in d for d in graph["dependencies"])
    # A head, when there is one, is part of the identity and survives the round trip.
    attached = modal_thought("быть", ("direct_object", noun("врач")))
    data = identity_graph(attached)
    assert [d.get("head") for d in data["dependencies"]] == [None, None, "action_complement"]
    assert thought_from_data(json.loads(json.dumps(data))) == attached
    assert thought_from_data(json.loads(json.dumps(graph))) == flat


def test_a_head_does_not_change_the_proposition_key():
    attached = modal_thought("стать", ("direct_object", noun("пилот", "instrumental")))
    flat = replace(attached, dependencies=tuple(
        replace(d, head=None) for d in attached.dependencies))
    assert proposition_key(attached) == proposition_key(flat)


def test_message_plan_key_ignores_an_absent_head():
    plan = MessagePlan(ReaderGoal("respond", None, (), None, False),
                       ClauseThought(ConceptLexeme("читать", "читать", "verb"), ()))
    assert plan.key == "f158ac0152f06d7f4c49c412"  # stage-3 value, unchanged


# ------------------------------------------------------------------ government by the own head

@pytest.mark.parametrize(("nested", "lemma", "expected"), [
    ("быть", "врач", "Я хочу быть врачом."),
    ("стать", "пилот", "Я хочу стать пилотом."),
    ("казаться", "кот", "Я хочу казаться котом."),
    ("являться", "художник", "Я хочу являться художником."),
    ("быть", "дерево", "Я хочу быть деревом."),
])
def test_an_argument_of_an_embedded_linking_verb_takes_the_case_of_that_verb(
        nested, lemma, expected):
    # No case in the plan: the embedded verb alone decides (it cannot govern the accusative).
    assert say(modal_thought(nested, ("direct_object", noun(lemma)))) == [expected]


def test_a_transitive_embedded_verb_still_governs_the_accusative():
    assert say(modal_thought("съесть", ("direct_object", noun("торт")))) == ["Я хочу съесть торт."]
    assert say(modal_thought("купить", ("direct_object", noun("машина")))) == [
        "Я хочу купить машину."]


def test_a_lexical_profile_of_government_goes_before_the_default():
    # «бояться» governs the genitive; the profile of the embedded verb wins over «instrumental».
    assert say(modal_thought("бояться", ("direct_object", noun("собака")))) == [
        "Я хочу бояться собаки."]


def test_a_case_fixed_by_the_plan_is_a_hard_constraint():
    thought = modal_thought("быть", ("direct_object", noun("врач", "nominative")))
    assert say(thought) == ["Я хочу быть врач."]


@pytest.mark.parametrize("pattern_roles", [
    ("subject", "predicate", "direct_object", "action_complement"),
    ("direct_object", "subject", "predicate", "action_complement"),
])
def test_an_argument_of_the_embedded_verb_stands_behind_that_verb_in_any_learned_order(
        pattern_roles):
    # «хотел бы собаку быть» is not a clause: whatever order a pattern gives the roles.
    pattern = DependencyPattern(pattern_roles, confidence=0.99, evidence_count=50,
                                predicate_class="modal", required_roles=pattern_roles)
    thought = modal_thought("быть", ("direct_object", noun("собака")), mood="conditional")
    texts = [s.text for s in realize_thought(thought, (pattern,), limit=6)]
    assert texts and all("быть собакой" in t for t in texts), texts


def test_without_a_head_the_old_flat_order_is_unchanged():
    # The opt-in link changes nothing for a thought that has none (the old behaviour is kept).
    pattern = DependencyPattern(("subject", "predicate", "direct_object", "action_complement"),
                                confidence=0.99, evidence_count=50, predicate_class="modal",
                                required_roles=("subject", "predicate", "action_complement"))
    flat = modal_thought("быть", ("direct_object", noun("собака", "instrumental")), head=None)
    assert any(s.role_order.index("direct_object") < s.role_order.index("action_complement")
               for s in realize_thought(flat, (pattern,), limit=6))


# ------------------------------------------------------------------ likeness «как X»

def parse(text, **kwargs):
    return parse_proposition(analyze_structured_utterance(text), allow_question=True,
                             tree=True, **kwargs)


@pytest.mark.parametrize(("text", "nested", "noun_lemma", "surface"), [
    ("ты хочешь быть как пилот?", "быть", "пилот", "Я хочу быть как пилот."),
    ("ты хочешь стать как врач?", "стать", "врач", "Я хочу стать как врач."),
    ("ты хочешь быть как кошка?", "быть", "кошка", "Я хочу быть как кошка."),
    ("ты хочешь быть как дерево?", "быть", "дерево", "Я хочу быть как дерево."),
])
def test_how_x_becomes_a_comparison_attached_to_the_embedded_verb(
        text, nested, noun_lemma, surface):
    thought = parse(text)
    assert thought is not None
    roles = {d.role: d for d in thought.dependencies}
    assert roles["action_complement"].argument.head.lemma == nested
    assert roles["comparison"].argument.head.lemma == noun_lemma
    assert roles["comparison"].head == "action_complement"
    assert roles["comparison"].argument.grammatical_case == "nominative"
    assert "direct_object" not in roles
    assert say(thought) == [surface]


def test_a_comparison_never_becomes_the_object_of_the_modal_in_the_default_reading():
    # The default reader is unchanged: the old wrong reading stays for the old callers.
    default = parse_proposition(
        analyze_structured_utterance("ты хочешь быть как пилот?"), allow_question=True)
    assert default is not None
    assert not any(d.role == "comparison" for d in default.dependencies)


def test_a_proper_name_after_how_stays_nominative_and_capitalised():
    thought = parse("ты хочешь быть как Гитлер?")
    assert say(thought) == ["Я хочу быть как Гитлер."]


@pytest.mark.parametrize("text", [
    "а как ты спишь?", "ты знаешь как это делается?", "знаешь как дела?",
    "как ты думаешь?", "ты не знаешь как мне быть?",
])
def test_a_question_about_manner_is_not_a_comparison(text):
    thought = parse(text)
    assert thought is None or not any(d.role == "comparison" for d in thought.dependencies)


def test_a_comparison_of_an_ordinary_verb_hangs_on_the_predicate():
    thought = parse("ты бегаешь как заяц?")
    comparison = next(d for d in thought.dependencies if d.role == "comparison")
    assert comparison.head is None and comparison.argument.head.lemma == "заяц"
    assert say(thought) == ["Я бегаю как заяц."]


def test_the_attached_object_of_the_embedded_verb_keeps_the_head_in_the_tree_reading():
    thought = parse("ты хотел бы быть собакой?", irrealis=True)
    attached = [d for d in thought.dependencies if d.head == "action_complement"]
    assert [d.argument.head.lemma for d in attached] == ["собака"]
    assert say(thought) == ["Я хотел бы быть собакой."]


# ------------------------------------------------------------------ the elided question

def test_an_elided_question_takes_the_complement_of_the_same_verb_from_an_earlier_turn():
    elided = parse("а хотел бы быть?", irrealis=True, fallback_subject=SELF)
    assert not any(d.head for d in elided.dependencies)
    completed = complete_nested_complement(
        elided, ("ты хочешь быть как пилот?", "привет"))
    assert [d.role for d in completed.dependencies if d.head] == ["comparison"]
    assert "elliptical_complement" in completed.evidence
    assert say(completed) == ["Я хотел бы быть как пилот."]


def test_an_earlier_turn_with_another_verb_or_none_adds_nothing():
    elided = parse("а хотел бы быть?", irrealis=True, fallback_subject=SELF)
    assert complete_nested_complement(elided, ("ты хочешь стать как пилот?",)) == elided
    assert complete_nested_complement(elided, ()) == elided
    assert complete_nested_complement(elided, ("привет", "как дела?")) == elided


# ------------------------------------------------------------------ thought_to_tree

def by_role(tree, role):
    return [n for n in tree["nodes"] if n["role"] == role]


def test_tree_of_a_modal_clause_has_heads_and_no_target_features():
    tree = thought_to_tree(parse("ты хочешь быть как кошка?"))
    json.dumps(tree, ensure_ascii=False)  # JSON-safe
    assert tree["version"] == 1 and tree["kind"] == "clause"
    nodes = {n["id"]: n for n in tree["nodes"]}
    assert nodes[0]["role"] == "predicate" and nodes[0]["head"] is None
    nested = by_role(tree, "action_complement")[0]
    comparison = by_role(tree, "comparison")[0]
    assert nested["head"] == 0 and comparison["head"] == nested["id"]
    assert comparison["marker"] == "как"
    assert comparison["lex"]["gender"] == "feminine"
    assert nodes[0]["lex"]["transitivity"] == "transitive"
    assert nested["lex"]["transitivity"] == "intransitive"
    for node in tree["nodes"]:
        assert "case" not in node and "person" not in node and "order" not in node
        assert not {"case", "person", "form"} & set(node["lex"])
    clause = tree["clause"]
    assert clause["move"] == "statement" and clause["mood"] == "indicative"
    assert clause["negated"] is False and clause["subject_elided"] is False


def test_tree_lists_nodes_in_a_canonical_order_not_the_order_of_the_plan():
    a = modal_thought("быть", ("direct_object", noun("врач")), ("location", noun("дом")))
    b = replace(a, dependencies=tuple(reversed(a.dependencies)))
    assert [n["lemma"] for n in thought_to_tree(a)["nodes"]] == [
        n["lemma"] for n in thought_to_tree(b)["nodes"]]
    # embedded argument is listed after its head in both listings, whatever the plan's order
    for plan in (a, b):
        tree = thought_to_tree(plan)
        ids = {n["role"]: n["id"] for n in tree["nodes"]}
        assert tree["nodes"][ids["direct_object"]]["head"] == ids["action_complement"]
        assert ids["action_complement"] < ids["direct_object"]


def test_tree_marks_mood_negation_focus_polarity_and_the_fixed_case():
    plan = ClauseThought(
        verb("видеть", ("mood", "conditional")),
        (SemanticDependency("subject", SELF),
         SemanticDependency("direct_object", ConceptArgument(
             ConceptLexeme("кто", "кто", "pronoun"), grammatical_case="accusative"))),
        clause_kind="question", negated=True, focus_role="direct_object",
        response_polarity=False, predicate_class="lexical")
    tree = thought_to_tree(plan)
    clause = tree["clause"]
    assert clause["move"] == "slot_question"
    assert clause["focus"] == by_role(tree, "direct_object")[0]["id"]
    assert clause["mood"] == "conditional" and clause["negated"] is True
    assert clause["negation_node"] == 0 and clause["response_polarity"] == "no"
    assert by_role(tree, "direct_object")[0]["case_constraint"] == "accusative"


def test_tree_carries_the_pro_drop_of_the_subject_and_a_preposition_of_the_plan():
    elided = ConceptArgument(ConceptLexeme(
        "я", "я", "pronoun", (("elided", "true"),)), grammatical_case="nominative")
    plan = ClauseThought(verb("жить"), (
        SemanticDependency("subject", elided),
        SemanticDependency("location", noun("лес", preposition="в"))), predicate_class="lexical")
    tree = thought_to_tree(plan)
    assert tree["clause"]["subject_elided"] is True
    assert by_role(tree, "subject")[0]["elided"] is True
    assert by_role(tree, "location")[0]["preposition"] == "в"


def test_tree_of_a_message_links_clauses_and_never_invents_words():
    claim = modal_thought("быть", ("direct_object", noun("кот")))
    reason = ClauseThought(verb("любить"), (
        SemanticDependency("subject", SELF), SemanticDependency("direct_object", noun("рыба"))),
        predicate_class="lexical")
    plan = MessagePlan(ReaderGoal("respond", None, (), None, False), claim, reason=reason,
                       follow_up=reason, speech_act="opinion")
    tree = thought_to_tree(plan)
    assert tree["kind"] == "message" and tree["speech_act"] == "opinion"
    assert [c["link"]["type"] for c in tree["clauses"]] == ["main", "cause", "coordination"]
    assert [c["id"] for c in tree["clauses"]] == [0, 1, 2]
    lemmas = {n["lemma"] for c in tree["clauses"] for n in c["nodes"]}
    assert lemmas == {"хотеть", "я", "быть", "кот", "любить", "рыба"}  # only the plan's words


def test_tree_is_deterministic():
    thought = parse("ты хочешь стать как дерево?")
    assert json.dumps(thought_to_tree(thought)) == json.dumps(thought_to_tree(thought))
