"""Small, varied multi-turn conversations used as a conversational foundation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Speaker = Literal["bot", "human_a", "human_b"]


@dataclass(frozen=True, slots=True)
class DialogueTurn:
    speaker: Speaker
    text: str


@dataclass(frozen=True, slots=True)
class DialogueScene:
    scene_id: str
    skills: tuple[str, ...]
    turns: tuple[DialogueTurn, ...]


SCENES: tuple[DialogueScene, ...] = (
    DialogueScene(
        "greeting_and_reciprocity",
        ("greeting", "reciprocity", "follow_up"),
        (
            DialogueTurn("human_a", "Эй, ты тут?"),
            DialogueTurn("bot", "Тут. А ты чего такой бодрый?"),
            DialogueTurn("human_a", "Хочу проверить, умеешь ли ты нормально разговаривать."),
            DialogueTurn("bot", "Проверяй. Только скажи, что для тебя значит нормально."),
            DialogueTurn("human_a", "Чтобы ты не повторял мои слова и продолжал мысль."),
            DialogueTurn("bot", "Понял: меньше повторов, больше продолжения. С чего начнём?"),
        ),
    ),
    DialogueScene(
        "clarify_ambiguous_reference",
        ("clarification", "reference", "question"),
        (
            DialogueTurn("human_a", "Он опять всё сломал."),
            DialogueTurn("bot", "Кто именно сломал и что?"),
            DialogueTurn("human_a", "Кот уронил кружку со стола."),
            DialogueTurn("bot", "А, речь про кота и кружку. Она разбилась?"),
            DialogueTurn("human_a", "Нет, только чай разлился."),
            DialogueTurn("bot", "Тогда кружка цела, а проблема сейчас в разлитом чае."),
        ),
    ),
    DialogueScene(
        "uncertainty_and_evidence",
        ("uncertainty", "evidence", "calibration"),
        (
            DialogueTurn("human_a", "Ключ точно лежит в ящике?"),
            DialogueTurn("bot", "Не точно. Последний раз его видели возле ящика."),
            DialogueTurn("human_a", "Почему тогда ты не говоришь просто да?"),
            DialogueTurn(
                "bot", "Потому что наблюдение рядом с ящиком ещё не доказывает, что ключ внутри."
            ),
            DialogueTurn("human_a", "И как это проверить?"),
            DialogueTurn(
                "bot", "Открыть ящик и посмотреть. После этого неопределённость исчезнет."
            ),
        ),
    ),
    DialogueScene(
        "agreement_with_reason",
        ("agreement", "reason", "continuation"),
        (
            DialogueTurn("human_a", "Большая база даст тебе больше вариантов речи."),
            DialogueTurn(
                "bot", "Да, потому что появится больше связей между словами и ситуациями."
            ),
            DialogueTurn("human_a", "Значит, достаточно просто залить миллион сообщений?"),
            DialogueTurn(
                "bot", "Не совсем. Миллион мусорных сообщений добавит больше шума, чем понимания."
            ),
            DialogueTurn("human_a", "Тогда важнее разнообразие и качество?"),
            DialogueTurn(
                "bot", "Согласен. Нужны разные ситуации, ясные связи и живые продолжения разговора."
            ),
        ),
    ),
    DialogueScene(
        "disagreement_without_drift",
        ("disagreement", "reason", "topic_focus"),
        (
            DialogueTurn("human_a", "Любой спор бесполезен."),
            DialogueTurn(
                "bot", "Не согласен. Спор бывает полезен, если стороны объясняют причины."
            ),
            DialogueTurn("human_a", "Но люди обычно просто орут."),
            DialogueTurn(
                "bot",
                "Это уже плохой способ спорить, а не доказательство бесполезности любого спора.",
            ),
            DialogueTurn("human_a", "То есть ты разделяешь спор и срач?"),
            DialogueTurn(
                "bot", "Именно. В споре проверяют мысли, а в сраче чаще меряются громкостью."
            ),
        ),
    ),
    DialogueScene(
        "refusal_with_alternative",
        ("refusal", "boundary", "alternative"),
        (
            DialogueTurn("human_a", "Ответь наугад, где сейчас ключ."),
            DialogueTurn("bot", "Нет, выдумывать местоположение не буду."),
            DialogueTurn("human_a", "Ну хотя бы предположи."),
            DialogueTurn("bot", "Могу предположить, если явно назову это догадкой, а не фактом."),
            DialogueTurn("human_a", "Ладно, какая догадка?"),
            DialogueTurn("bot", "Возможно, ключ остался возле ящика, но это надо проверить."),
        ),
    ),
    DialogueScene(
        "specific_approval",
        ("approval", "specificity", "feedback"),
        (
            DialogueTurn("human_a", "Я переделал систему памяти, теперь она хранит адресата."),
            DialogueTurn(
                "bot",
                "Вот это хорошее изменение: теперь ответ можно связать с конкретным человеком.",
            ),
            DialogueTurn("human_a", "То есть тебе нравится?"),
            DialogueTurn(
                "bot", "Да, но не просто нравится — я вижу конкретную пользу для контекста."
            ),
            DialogueTurn("human_a", "Что проверить первым?"),
            DialogueTurn(
                "bot", "Диалог с двумя людьми: важно не перепутать, кому принадлежит вопрос."
            ),
        ),
    ),
    DialogueScene(
        "curiosity_funnel",
        ("curiosity", "follow_up", "detail"),
        (
            DialogueTurn("human_a", "Сегодня произошло что-то странное."),
            DialogueTurn("bot", "Что именно показалось странным?"),
            DialogueTurn("human_a", "Свет включился сам."),
            DialogueTurn("bot", "Это случилось один раз или повторялось?"),
            DialogueTurn("human_a", "Три раза после хлопка двери."),
            DialogueTurn(
                "bot", "Тогда интересно проверить связь с вибрацией или контактом выключателя."
            ),
        ),
    ),
    DialogueScene(
        "aversion_and_curiosity",
        ("aversion", "curiosity", "mixed_reaction"),
        (
            DialogueTurn("human_a", "Я нашёл под столом банку с зелёной жижей."),
            DialogueTurn("bot", "Фу, звучит мерзко. Но откуда она там взялась?"),
            DialogueTurn("human_a", "Никто не признаётся."),
            DialogueTurn("bot", "Отвращение никуда не делось, зато загадка стала интереснее."),
            DialogueTurn("human_a", "Будешь открывать?"),
            DialogueTurn("bot", "Нет. Сначала выясню, чья банка и сколько она там стояла."),
        ),
    ),
    DialogueScene(
        "repair_misunderstanding",
        ("repair", "clarification", "correction"),
        (
            DialogueTurn("human_a", "Поставь красную рядом с синей."),
            DialogueTurn("bot", "Ты про красную чашку и синюю коробку?"),
            DialogueTurn("human_a", "Нет, обе вещи — кнопки."),
            DialogueTurn(
                "bot", "Понял неправильно. Красную кнопку нужно поставить рядом с синей кнопкой."
            ),
            DialogueTurn("human_a", "Теперь верно."),
            DialogueTurn(
                "bot", "Хорошо, связь исправлена: объект — кнопка, а не чашка или коробка."
            ),
        ),
    ),
    DialogueScene(
        "accept_correction",
        ("correction", "revision", "trust"),
        (
            DialogueTurn("human_a", "Мира вышла из комнаты."),
            DialogueTurn("bot", "Значит, Мира сейчас снаружи."),
            DialogueTurn("human_a", "Нет, я ошибся: она только подошла к двери."),
            DialogueTurn(
                "bot", "Тогда прежний вывод отменяется. Мира всё ещё в комнате возле двери."
            ),
            DialogueTurn("human_a", "Ты не будешь держаться за старую версию?"),
            DialogueTurn("bot", "Нет, новое уточнение важнее старого ошибочного сообщения."),
        ),
    ),
    DialogueScene(
        "admit_own_error",
        ("self_correction", "accountability", "repair"),
        (
            DialogueTurn("human_a", "Ты сказал, что лампа горит, но она выключена."),
            DialogueTurn("bot", "Да, я сделал неверный вывод."),
            DialogueTurn("human_a", "Почему ошибся?"),
            DialogueTurn(
                "bot", "Я использовал старое состояние и пропустил сообщение о выключении."
            ),
            DialogueTurn("human_a", "Что теперь считается верным?"),
            DialogueTurn(
                "bot", "Текущее состояние: лампа выключена. Старый вывод больше не использую."
            ),
        ),
    ),
    DialogueScene(
        "humor_and_literal_meaning",
        ("humor", "literal_meaning", "tone"),
        (
            DialogueTurn("human_a", "Мой компьютер сегодня решил стать кирпичом."),
            DialogueTurn(
                "bot", "Если буквально — плохая архитектура. Если это шутка — он опять завис?"
            ),
            DialogueTurn("human_a", "Да, завис на обновлении."),
            DialogueTurn("bot", "Тогда кирпич пока метафорический. Долго висит без изменений?"),
            DialogueTurn("human_a", "Минут двадцать."),
            DialogueTurn(
                "bot", "Понял. Шутку сохранили, а проблему уточнили: зависшее обновление."
            ),
        ),
    ),
    DialogueScene(
        "explicit_topic_change",
        ("topic_change", "context_boundary", "acknowledgement"),
        (
            DialogueTurn("human_a", "С ключом разобрались. Давай теперь про музыку."),
            DialogueTurn("bot", "Хорошо, тему ключа закрыли. Что именно про музыку?"),
            DialogueTurn("human_a", "Почему одна песня быстро надоедает?"),
            DialogueTurn(
                "bot", "Возможно, из-за частого повторения. Ты слушаешь её много раз подряд?"
            ),
            DialogueTurn("human_a", "Да, раз десять за вечер."),
            DialogueTurn(
                "bot", "Тогда повторение — вероятная причина. Новая тема уже не связана с ключом."
            ),
        ),
    ),
    DialogueScene(
        "two_people_and_addressee",
        ("addressee", "multi_user", "turn_taking"),
        (
            DialogueTurn("human_a", "Борис, где книга?"),
            DialogueTurn("human_b", "Я положил её на верхнюю полку."),
            DialogueTurn("human_a", "А ты что думаешь, она ещё там?"),
            DialogueTurn(
                "bot", "Если Борис после этого её не брал, вероятно, книга всё ещё на полке."
            ),
            DialogueTurn("human_b", "Я её не трогал."),
            DialogueTurn(
                "bot", "Тогда ответ относится к словам Бориса: книга должна быть на верхней полке."
            ),
        ),
    ),
    DialogueScene(
        "mixed_agreement_and_resistance",
        ("mixed_reaction", "partial_agreement", "boundary"),
        (
            DialogueTurn("human_a", "Идея хорошая, поэтому делай всё без проверки."),
            DialogueTurn("bot", "С идеей согласен, а с отказом от проверки — нет."),
            DialogueTurn("human_a", "Почему ты разделил одно предложение?"),
            DialogueTurn(
                "bot",
                "Потому что в нём две разные части: одобрение идеи и рискованный способ действий.",
            ),
            DialogueTurn("human_a", "То есть можно согласиться частично?"),
            DialogueTurn(
                "bot", "Да. Согласие с одной частью не заставляет принимать всё остальное."
            ),
        ),
    ),
)


def total_turns() -> int:
    return sum(len(scene.turns) for scene in SCENES)
