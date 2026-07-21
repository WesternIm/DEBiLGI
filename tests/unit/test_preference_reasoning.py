from debilgi.language.preference_reasoning import (
    appraise_preference,
    preference_position_thought,
    preference_reason_thought,
)
from debilgi.language.productive_nlg import realize_thought


def test_known_properties_create_a_position_and_a_reusable_reason() -> None:
    appraisal = appraise_preference(
        "небо",
        (
            "Небо — пространство над Землёй; при взгляде видно панораму "
            "в направлении космоса.",
        ),
    )

    assert appraisal.stance == "like"
    assert appraisal.reason_lemma in {"пространство", "панорама", "космос"}
    position = realize_thought(preference_position_thought(appraisal))[0].text
    reason = realize_thought(
        preference_reason_thought(appraisal, correcting_negative_premise=True)
    )[0].text
    assert position == "Да, я люблю небо."
    assert reason.startswith("Нет, ")
    assert reason.endswith("привлекает меня.")


def test_unknown_properties_do_not_create_a_random_yes_or_no_position() -> None:
    appraisal = appraise_preference("фырбуль")

    assert appraisal.stance == "undecided"
    assert appraisal.reason_lemma is None
    assert preference_reason_thought(appraisal) is None
    surface = realize_thought(preference_position_thought(appraisal))[0].text
    assert "отношение" in surface
    assert "фырбул" in surface


def test_newly_learned_definition_can_ground_a_transferable_preference() -> None:
    appraisal = appraise_preference(
        "эмодзи",
        (
            "Язык идеограмм и смайликов, используемый в электронных сообщениях; "
            "сами пиктограммы.",
        ),
    )

    assert appraisal.stance == "like"
    assert appraisal.dimension == "expressiveness"
    assert appraisal.reason_lemma in {
        "идеограмма",
        "смайлик",
        "сообщение",
        "пиктограмма",
        "язык",
    }
    assert appraisal.topic == "эмодзи"
    assert realize_thought(preference_position_thought(appraisal))[0].text == (
        "Да, я люблю эмодзи."
    )
