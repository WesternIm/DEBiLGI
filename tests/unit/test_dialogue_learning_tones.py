import pytest

from debilgi.language.dialogue_learning import evaluation_tone_forms


def _rows():
    pairs = (("я купил машину", "вау"), ("я выиграл конкурс", "вау"), ("ты кто?", "вау"),
             ("я сдал экзамен", "вау"))
    return [{"prompt": p, "answer": a, "prompt_actor_id": 10 + i % 3,
             "answer_event_id": f"a{i}", "prompt_event_id": f"p{i}"}
            for i, (p, a) in enumerate(pairs)]


@pytest.mark.skip(reason="П.3 отключён (гейты 117): stage417_p3_acknowledgment_filter.patch")
def test_agreement_does_not_borrow_a_sound_people_never_said_after_agreement():
    rows = _rows()
    # The sound is witnessed after other acts, so an unknown evaluation may borrow it ...
    assert evaluation_tone_forms("странная штука", rows)
    # ... but never an agreement: nobody in the witnesses answered one this way.
    assert evaluation_tone_forms("окей", rows, reaction="acknowledgment") == ()
    assert evaluation_tone_forms("окей", rows, reaction="acknowledgment", recovery=True) == ()
