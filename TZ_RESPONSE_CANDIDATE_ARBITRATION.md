# Техническое задание: свободная генерация ответов через двухуровневый арбитраж кандидатов

## 1. Цель

Снизить заскриптованность DEBiLGI без потери фактической точности, связности диалога и
обучения на обратной связи.

Вместо выбора готового ответа по порядку веток `if/elif/return` система должна:

1. один раз разобрать входящее сообщение и собрать общий контекст;
2. запросить у всех применимых источников смысловые предложения ответа;
3. проверить обязательные инварианты;
4. численно оценить все допустимые смысловые предложения;
5. для лучших смысловых предложений создать несколько вариантов формулировки;
6. численно оценить варианты формулировки;
7. выбрать и отправить победителя;
8. сохранить весь арбитраж и связать его с последующим фидбеком;
9. постепенно калибровать веса выбора на подтверждённом фидбеке.

Арбитраж не должен публиковать в Discord внутренние оценки, причины отбраковки, процесс
поиска, список кандидатов или служебные состояния.

## 2. Ограничения проекта

- Проект остаётся локальным символическим ботом без подключения генеративной нейросети.
- Новые зависимости не требуются: реализация использует Python 3.11, SQLite и существующие
  языковые компоненты.
- Старые миграции не редактируются. Изменения схемы добавляются новой миграцией после `026`.
- Обработка должна оставаться детерминированной при одинаковом входе, версии весов и seed.
- Изменение выбора вводится через `shadow`-режим, а не одномоментной заменой работающего пути.
- Существующие факты, identity-контекст, reply-chain, working memory, словари, граф, reflection,
  self-dataset и опыт должны переиспользоваться, а не дублироваться.

## 3. Фактическое состояние текущего кода

### 3.1. Где уже есть мультивариантность

`MemoryPhraseGenerator` в `src/debilgi/language/memory_based.py` уже:

- строит от 20 до 96 поверхностных кандидатов;
- добавляет кандидаты из knowledge facts;
- оценивает тему, grounding, новизну, повторяемость, связность, грамматику, длину,
  соответствие речевому акту, социальной реакции и usage hints;
- выбирает максимальный балл.

Этот механизм работает только внутри одной графово-марковской ветки. Он не сравнивает себя с
identity, short act, contextual learning, lexical retrieval, dialogue rules, recovery и cache.

### 3.2. Где верхнеуровневый выбор остаётся порядковым

`EventListener.on_message` в `src/debilgi/discord/events.py` выбирает по порядку:

1. bound explanation;
2. bound short act;
3. гарантированный ответ на обращение;
4. initiative;
5. отсутствие ответа.

`SafeMentionReplyService._generate_checked` в `src/debilgi/discord/replies.py` содержит ранние
победы:

- identity возвращается до остальных источников;
- stable cache возвращается до генерации;
- dialogue rule / lexical retrieval / social routine возвращаются до графа;
- low-confidence unknown немедленно переводит ответ в recovery;
- первый прошедший retry генератора становится ответом;
- после провала выполняется последовательная замена на recovery или fallback.

Следовательно, порядок вызовов сейчас фактически является скрытым приоритетом источников.

### 3.3. Контекстный гейт

`WorkingMemoryContextGate.evaluate` сейчас возвращает дискретное решение:

- `passed`;
- `replaced` готовой fallback-фразой;
- `deferred`.

Источники `fallback`, `short_act` и `contextual_learning` полностью обходят проверку темы.
Для реактивного ответа нерелевантный текст заменяется готовой фразой уже после генерации.

В целевой архитектуре реактивная релевантность должна стать числовым признаком кандидата.
Жёсткий `deferred` сохраняется только для планирования фоновой инициативы, поскольку это
ограничение времени отправки, а не качество формулировки.

### 3.4. Текущий фидбек

Фидбек уже привязывается к `response_message_id`, но используется неодинаково:

- `ShortSpeechActService` надёжно связывает подтверждение/отрицание с конкретным ходом бота;
- `SelfDatasetRepository` различает явное одобрение, исправление, отказ, реакции и обычное
  продолжение;
- `ExperienceConsolidator` считает любой прямой reply положительным `+0.90`, даже если это
  текстовое исправление;
- почти любая добавленная реакция, кроме небольшого списка отрицательных emoji, считается
  положительной;
- опыт изменяет веса токен-переходов, но не веса верхнеуровневого выбора источников;
- reflection обучает длину, осторожность, exploration и ширину поиска, но не знает признаков
  проигравших кандидатов.

Для обучения арбитра нужен единый атрибутор фидбека. Обычное продолжение разговора не должно
считаться сильным подтверждением правильности победителя.

### 3.5. Что именно создаёт ощущение скрипта

Основная проблема не в наличии условий как таковых. Условия применимости всё равно нужны.
Скриптовость создают функции, одновременно выполняющие три задачи:

1. распознают конкретный случай;
2. выбирают решение без конкуренции;
3. возвращают полностью написанную фразу.

Примеры находятся в `language/recovery.py`, `language/short_acts.py`, `language/entities.py`,
`discord/replies.py` и replacement-ветке `language/context_gate.py`.

## 4. Основной архитектурный принцип

Система строится как двухуровневый арбитраж:

```text
Event + Understanding + Working Focus
    -> ContentProposalProvider[]
    -> hard validity checks
    -> content feature extraction
    -> content arbitration
    -> top content proposals
    -> SurfaceRealizer[]
    -> surface candidates
    -> hard surface checks
    -> surface feature extraction
    -> surface arbitration
    -> PreparedReply
```

Первый уровень решает, **что сказать**. Второй уровень решает, **как это сформулировать**.

Если оставить только один уровень, выбирающий готовые строки, бот продолжит выбирать лучший
из нескольких скриптов, но не станет свободнее в формулировках.

## 5. Термины и модели данных

### 5.1. `ResponseContentPlan`

Расширить или заменить текущий `ResponsePlan` структурой, содержащей:

```python
@dataclass(frozen=True, slots=True)
class ResponseContentPlan:
    speech_act: str
    response_goal: str
    target_id: int | None
    topic_ids: tuple[str, ...]
    key_points: tuple[str, ...]
    required_facts: tuple[str, ...]
    optional_facts: tuple[str, ...]
    unknowns: tuple[str, ...]
    tone: str
    certainty: float
    follow_up: str | None
```

`required_facts` — факты, которые нельзя потерять или исказить при свободной формулировке.
Например, имя Discord-пользователя, результат вычисления или точное найденное определение.

### 5.2. `ContentProposal`

```python
@dataclass(frozen=True, slots=True)
class ContentProposal:
    proposal_id: str
    source: str
    plan: ResponseContentPlan
    applicability: float
    source_confidence: float
    evidence_ids: tuple[str, ...]
    metadata: Mapping[str, object]
```

`source` должен быть нормализованным enum/строкой из ограниченного набора, а не произвольной
диагностической подписью.

Начальный набор источников:

- `discord_identity`;
- `contextual_learning`;
- `short_act`;
- `exact_archive`;
- `lexical_reference`;
- `knowledge_retrieval`;
- `semantic_memory`;
- `dialogue_act`;
- `graph_generation`;
- `recovery`;
- `cache`;
- `initiative`.

### 5.3. `SurfaceCandidate`

```python
@dataclass(frozen=True, slots=True)
class SurfaceCandidate:
    candidate_id: str
    proposal_id: str
    source: str
    renderer: str
    text: str
    seed: str
    metadata: Mapping[str, object]
```

### 5.4. `CandidateFeatures`

Все признаки нормализуются в диапазон `[0.0, 1.0]`. Неизвестный признак хранится как `None`,
а не как ложный ноль.

Положительные признаки:

- `applicability` — применимость источника к речевому акту;
- `context_relevance` — связь с working memory и reply-chain;
- `source_confidence` — подтверждённость источника;
- `speech_act_fit` — выполнение вопроса, команды, подтверждения и т. п.;
- `plan_coverage` — покрытие key points;
- `fact_grounding` — опора на required/optional facts;
- `topic_continuity` — сохранение текущей темы;
- `grammar_fit` — грамматическая связность;
- `coherence` — связность токен-переходов и предложения;
- `specificity` — наличие содержательного предмета и действия;
- `length_fit` — близость к целевой длине;
- `style_fit` — соответствие personality/reaction/reflection;
- `non_repetition` — отличие от последних ответов;
- `uncertainty_calibration` — соответствие уверенности формулировки уверенности плана.

Штрафные признаки:

- `contradiction_penalty`;
- `unsupported_certainty_penalty`;
- `prompt_echo_penalty`;
- `retrieval_dump_penalty`;
- `repetition_penalty`;
- `internal_leak_penalty`;
- `broken_surface_penalty`;
- `unrelated_tail_penalty`.

### 5.5. `CandidateScore` и `ArbitrationDecision`

```python
@dataclass(frozen=True, slots=True)
class CandidateScore:
    candidate_id: str
    features: CandidateFeatures
    weighted_components: Mapping[str, float]
    source_bias: float
    total: float
    eligible: bool
    rejection_reasons: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class ArbitrationDecision:
    decision_id: str
    event_id: str
    plan_id: str | None
    winner_id: str | None
    scores: tuple[CandidateScore, ...]
    weight_version: int
    minimum_score: float
    mode: str
    used_fallback: bool
```

## 6. Источники смысловых предложений

Каждый provider реализует единый протокол:

```python
class ContentProposalProvider(Protocol):
    async def propose(self, request: ArbitrationRequest) -> tuple[ContentProposal, ...]: ...
```

Provider имеет право вернуть пустой tuple. Это проверка применимости источника, а не выбор
победителя и не формирование окончательного приоритета.

### 6.1. Identity

Текущий `identity_reply` должен перестать возвращать готовую строку. Он создаёт план с:

- точной сущностью;
- требуемым display name / username;
- типом речевого акта `identify_entity`;
- `required_facts`, содержащим фактическое имя.

Поверхностный renderer создаёт несколько допустимых формулировок, сохраняя required facts.

### 6.2. Contextual learning

`ExplanationResolution.confirmation` преобразуется в смысловой план подтверждения усвоенного
значения. Существующий confirmation-текст может участвовать как один retrieval-кандидат, но не
должен быть единственной возможной формулировкой.

### 6.3. Short speech act

`ShortActResolution.reply_text()` заменяется на `to_content_plan()`.

Binding к конкретному предыдущему сообщению остаётся жёстким фактом. Вариативной становится
только формулировка подтверждения, отрицания, продолжения или уточняющего вопроса.

### 6.4. Retrieval

Разделить:

- точный подтверждённый archive answer;
- частичный archive retrieval;
- словарное определение;
- knowledge fact;
- semantic memory.

Каждый результат сохраняет собственную confidence и evidence ID. Полный точный ответ может
быть одним из surface-кандидатов, но не получает победу только из-за порядка кода.

### 6.5. Dialogue act

Текущие специальные реакции из `compose_dialogue_reply` и `compose_recovery` сначала
переносятся в смысловые планы:

- `answer_wellbeing`;
- `accept_or_reject_label`;
- `acknowledge_correction`;
- `answer_identity_relation`;
- `accept_rest_directive`;
- `request_missing_detail`;
- другие существующие речевые акты.

Распознавание может оставаться детерминированным. Из функций удаляется готовая авторская
фраза. Один акт может иметь несколько renderers и несколько вариантов поверхности.

### 6.6. Graph generation

`MemoryPhraseGenerator` получает новый публичный метод `generate_candidates(...)`, который:

- возвращает уникальные кандидаты и их внутренние признаки;
- не выбирает победителя самостоятельно в active-режиме;
- не делает ранний `return` для `exact_answer`;
- ограничивает число кандидатов общим бюджетом арбитража;
- сохраняет детерминированный seed.

Старый `generate()` временно остаётся адаптером для shadow-режима и обратной совместимости.

### 6.7. Cache

Stable cache перестаёт быть ранним `return`.

Допустимы два режима:

1. cached text участвует как кандидат с признаками `cache_age`, `past_reward`,
   `semantic_seed_match`;
2. полностью кэшируется решение только по ключу `(semantic_seed, weight_version,
   provider_version)`.

При изменении весов старое решение не должно бесконечно обходить новый арбитраж.

### 6.8. Recovery и fallback

Recovery — обычный смысловой provider с низкой или средней source confidence, но высокой
применимостью при недостатке данных.

Общий нейтральный fallback не участвует на равных с содержательными кандидатами. Он создаётся
только когда:

- нет eligible-кандидатов;
- максимальный итоговый score ниже `minimum_score`;
- все кандидаты превысили timeout или завершились ошибкой.

Fallback тоже строится из плана (`acknowledge_gap`, `request_specific_detail`), а не выбирается
из глобального счётчика готовых строк.

Транспортный/system failure остаётся отдельным аварийным путём и не обучает веса обычного
арбитража.

## 7. Жёсткие инварианты

Числовой score не заменяет проверки корректности. Кандидат исключается до ранжирования, если:

- текст пустой или превышает Discord/настроенные ограничения;
- содержит внутренние маркеры, технические placeholder-токены или диагностический текст;
- потерял обязательный факт;
- противоречит required fact;
- содержит незакрытые критические конструкции после repair;
- не соответствует обязательному типу результата команды;
- является точным эхом пользовательского сообщения без самостоятельного содержания;
- адресует действие не тому Discord-пользователю при однозначном entity context.

Topic overlap, длина, стилистика, обычная повторяемость и общая специфичность не являются
жёсткими запретами — они изменяют score.

## 8. Формула арбитража

Начальная формула:

```text
score(candidate) = source_bias[source]
                 + sum(weight[name] * positive_feature[name])
                 - sum(penalty_weight[name] * penalty_feature[name])
```

Требования:

- все признаки вычисляются одним `CandidateFeatureExtractor`, а не каждым provider по-своему;
- provider сообщает evidence/confidence, но не финальный score;
- веса и source biases загружаются из versioned profile;
- tie-break детерминирован через `stable_seed + candidate_id`;
- перед сравнением выполняется дедупликация одинаковых и почти одинаковых текстов;
- при одинаковой формулировке от нескольких источников evidence объединяется, а confidence не
  суммируется без ограничения;
- content и surface стадии имеют разные наборы весов;
- score и каждый weighted component сохраняются.

Начальные веса нужны для cold start, но не считаются вечными. Они задаются как baseline,
относительно которого работает ограниченное обучение.

## 9. Контекстный гейт в целевой архитектуре

`WorkingMemoryContextGate` разделяется на две функции:

1. `measure_relevance(...) -> ContextRelevance` — возвращает числовые признаки:
   direct concept match, graph-related match, reply-chain binding, active-topic coverage,
   recency;
2. `schedule_initiative(...)` — сохраняет `passed/deferred` для фоновых и инициативных
   сообщений.

Для реактивных ответов гейт не пишет replacement-текст и не выбирает winner. Его измерения
входят в `context_relevance` и `topic_continuity` каждого кандидата.

Специальный `_BOUND_SOURCES` удаляется после перехода на новый pipeline. Binding short act и
contextual learning выражается отдельным числовым признаком `reply_binding` и обязательными
фактами, а не полным обходом релевантности.

## 10. Поверхностная вариативность

Для каждого из top-N смысловых планов генерируется несколько поверхностей:

- retrieval surface — исходный подтверждённый текст, если он допустим;
- graph/Markov surface;
- learned reply-head surface;
- concise surface;
- balanced surface;
- expressive surface, если это допускает social reaction/personality.

Варианты обязаны сохранять:

- required facts;
- speech act;
- отрицание;
- адресата;
- certainty;
- follow-up intention.

Они могут менять:

- порядок частей;
- синтаксис;
- связки;
- степень краткости;
- допустимые слова-паразиты и эмоциональные маркеры;
- финальный уточняющий вопрос.

Нельзя обеспечивать вариативность простым случайным выбором из списка готовых полных фраз.
Допустимы небольшие лексические банки и выученные конструкции, если итог собирается из плана и
проходит общий surface arbitration.

## 11. Обучение весов

### 11.1. Единый атрибутор фидбека

Создать `FeedbackAttributor`, который один раз классифицирует событие относительно конкретного
`response_message_id`:

- `explicit_confirmation`: `+1.0`, высокая attribution confidence;
- `explicit_rejection/correction`: `-1.0`, высокая confidence;
- известная положительная reaction: `+0.8`;
- известная отрицательная reaction: `-0.8`;
- reaction removed: инверсия/отмена предыдущего сигнала;
- обычный direct reply без оценочной семантики: слабый сигнал не выше `+0.10`;
- продолжение разговора без reply: нейтральный сигнал для арбитражных весов;
- timeout/ignored: не более слабого отрицательного сигнала и только для отдельных
  response-mode признаков.

`ExperienceConsolidator`, `SelfDatasetRepository` и новый learner должны получать один и тот же
атрибутированный результат, чтобы один reply не считался одновременно `+0.90` и `-0.90`.

### 11.2. Что обучается

- content feature weights;
- surface feature weights;
- source biases;
- при достаточных данных — blend с per-user preference.

Токен-переходы продолжают обучаться существующим механизмом. Reflection продолжает менять
длину, осторожность и exploration. Новый learner не должен повторно менять эти же параметры.

### 11.3. Начальный scope

Первая версия обучает профиль на уровне `guild_id`.

Per-user слой добавляется только после достаточного количества надёжно атрибутированных
наблюдений и смешивается с guild profile по confidence. Это предотвращает создание шумного
профиля после пары сообщений.

### 11.4. Обновление

Рекомендуемый ограниченный pairwise update:

```text
delta = features(winner) - weighted_mean(features(top alternatives))
weight += learning_rate * reward * attribution_confidence * delta
```

Для отрицательной награды направление автоматически смещается в сторону лучших альтернатив.

Обязательные ограничения:

- обновлять только решение, реально связанное с feedback event;
- не обучаться по system failure;
- не обучаться по неатрибутированному продолжению;
- learning rate по умолчанию не выше `0.03`;
- одно обновление ограничено max delta;
- положительные веса не могут стать отрицательными;
- penalty weights не могут сменить знак;
- source bias имеет отдельный ограниченный диапазон;
- профиль хранит baseline, observation count, confidence и version;
- предусмотрено медленное возвращение к baseline при устаревании данных;
- каждое обновление идемпотентно по `feedback_event_id`;
- возможен rollback на предыдущую версию профиля.

### 11.5. Exploration

В active-режиме winner обычно равен max score. Опциональный exploration допускается только:

- после накопления минимального числа наблюдений;
- среди eligible-кандидатов без hard violations;
- среди near-tie кандидатов;
- с малой вероятностью;
- с обязательным логированием `selection_reason=exploration`.

В первой активной версии exploration выключен.

## 12. Хранилище

Добавить новую миграцию `027_response_candidate_arbitration.sql`.

### 12.1. `language_arbitration_decisions`

- `decision_id` primary key;
- `event_id` unique;
- `plan_id` nullable;
- `guild_id`, `channel_id`, `user_id`;
- `mode` (`shadow`, `active`);
- `winner_candidate_id` nullable;
- `legacy_winner_text_hash` nullable;
- `minimum_score`;
- `weight_profile_id`;
- `weight_version`;
- `candidate_count`;
- `eligible_count`;
- `used_fallback`;
- `selection_reason`;
- `latency_ms`;
- `created_at`.

### 12.2. `language_response_candidates`

- `candidate_id` primary key;
- `decision_id` foreign key;
- `proposal_id`;
- `source`;
- `renderer`;
- `content_plan_json`;
- `text` или ограниченный диагностический текст;
- `text_hash`;
- `features_json`;
- `weighted_components_json`;
- `source_confidence`;
- `total_score`;
- `eligible`;
- `rejection_reasons_json`;
- `selected`;
- `generation_order` только для диагностики, не для приоритета;
- `created_at`.

Все кандидаты сохраняются, но их общее число ограничено конфигурацией. Для candidate text
добавляется retention cleanup, поскольку остальные таблицы проекта уже сохраняют исходные
реплики и ответы.

### 12.3. `language_arbitration_weight_profiles`

- `profile_id` primary key;
- `scope_type` (`global`, `guild`, `user`);
- `scope_id`;
- `stage` (`content`, `surface`);
- `weights_json`;
- `source_biases_json`;
- `baseline_json`;
- `observations`;
- `confidence`;
- `version`;
- `created_at`, `updated_at`;
- unique `(scope_type, scope_id, stage, version)`.

### 12.4. `language_arbitration_feedback`

- `feedback_event_id` primary key;
- `decision_id` foreign key;
- `experience_id` nullable;
- `feedback_kind`;
- `reward`;
- `attribution_confidence`;
- `applied_profile_id` nullable;
- `old_version`, `new_version` nullable;
- `weight_delta_json`;
- `observed_at`, `applied_at`.

### 12.5. Существующие модели

`PreparedReply` расширить полями:

- `decision_id`;
- `winner_candidate_id`;
- `actual_candidate_count`;
- `arbitration_mode`;
- `weight_version`.

Текущее `candidate_count` должно означать фактическое число уникальных кандидатов, а не
запрошенный personality search width.

Все обычные отправленные победители, включая recovery и neutral fallback, должны иметь
`actual_candidate_count >= 1`. Аварийный transport/system fallback может не создавать обычный
experience, но должен создавать диагностическую запись failure.

## 13. Логирование и диагностика

Добавить logger `debilgi.response_arbitration` и файл `data/response-arbitration.log` через
существующий rotating logging.

Для каждого решения логировать:

- decision/event ID;
- режим;
- weight profile/version;
- количество generated/deduplicated/eligible кандидатов;
- источник, renderer и итоговый score каждого кандидата;
- значения всех признаков;
- weighted components;
- hard rejection reasons;
- winner и selection reason;
- fallback reason;
- latency каждого provider и всего арбитража;
- ошибку отдельного provider без остановки остальных.

Лог не содержит токены Discord, скрытые переменные окружения и chain-of-thought. Допустимо
логировать ограниченный однострочный candidate text либо hash + ссылку на строку в SQLite.

Добавить административную диагностику:

- последние решения;
- перекос побед по источникам;
- средний score по источникам;
- доля fallback;
- hard rejection counts;
- текущие веса и версии;
- средний/95-й percentile latency;
- shadow disagreement rate между legacy и новым winner.

## 14. Конфигурация

Добавить настройки:

```text
DEBILGI_ARBITRATION_MODE=off|shadow|active
DEBILGI_ARBITRATION_MIN_SCORE=0.45
DEBILGI_ARBITRATION_MAX_CONTENT_PROPOSALS=12
DEBILGI_ARBITRATION_TOP_CONTENT_PROPOSALS=4
DEBILGI_ARBITRATION_SURFACES_PER_PROPOSAL=4
DEBILGI_ARBITRATION_MAX_TOTAL_CANDIDATES=48
DEBILGI_ARBITRATION_PROVIDER_TIMEOUT_MS=2200
DEBILGI_ARBITRATION_LEARNING_ENABLED=false
DEBILGI_ARBITRATION_LEARNING_RATE=0.03
DEBILGI_ARBITRATION_MIN_FEEDBACK_OBSERVATIONS=20
DEBILGI_ARBITRATION_MAX_WEIGHT_DELTA=0.05
DEBILGI_ARBITRATION_EXPLORATION_RATE=0.0
DEBILGI_ARBITRATION_CANDIDATE_RETENTION_DAYS=30
```

Все значения валидируются в `Settings.from_env()` и документируются в `.env.example`.

## 15. Параллельность, ошибки и бюджет времени

- Независимые providers запускаются через `asyncio.gather` с изоляцией исключений.
- Ошибка одного provider не отменяет остальных.
- Внешние reference/knowledge providers сохраняют существующие timeout и policy.
- Общий provider timeout ограничивается конфигурацией.
- Медленный provider может завершиться после дедлайна только для фонового обучения, но не
  менять уже отправленный winner.
- Дедупликация выполняется до дорогой полной surface-оценки.
- Верхний предел кандидатов обязателен, иначе логирование и score превратятся в бесконтрольный
  рост.

## 16. Shadow-режим

В `shadow`-режиме:

1. legacy pipeline формирует реальный ответ;
2. новый pipeline собирает и оценивает кандидатов;
3. в Discord отправляется только legacy answer;
4. сохраняются legacy winner, shadow winner и disagreement;
5. обучение весов отключено;
6. provider failures не влияют на отправку.

Переход в `active` разрешается после выполнения критериев:

- не менее 500 shadow decisions либо согласованный меньший тестовый объём;
- отсутствие hard-invariant regressions;
- fallback rate не хуже legacy более чем на 5 процентных пунктов;
- p95 latency укладывается в согласованный бюджет;
- disagreement вручную просмотрен по основным источникам;
- нет постоянного доминирования одного источника из-за масштаба признака.

## 17. Этапы реализации

### Этап 1. Модели и наблюдаемость без изменения поведения

- добавить модели кандидатов и решений;
- добавить миграцию 027 и repository;
- реализовать feature extractor;
- превратить context gate relevance в измеряемую структуру, сохранив старый `evaluate` как
  адаптер;
- добавить shadow orchestrator;
- логировать legacy и shadow выбор;
- не менять отправляемый ответ.

### Этап 2. Верхнеуровневые providers

- обернуть identity, short act, contextual learning, retrieval, dialogue rule, recovery, cache;
- удалить приоритет, зависящий от порядка provider-ов, в shadow path;
- вынести общий `ArbitrationRequest`, чтобы parsing/knowledge/context не повторялись каждым
  provider;
- обеспечить provider concurrency и timeout.

### Этап 3. Мультивариантная поверхность

- добавить `generate_candidates` в `MemoryPhraseGenerator`;
- перевести наиболее заметные готовые ответы из `recovery.py` и `short_acts.py` в content plans;
- реализовать concise/balanced/expressive renderers;
- валидировать required facts и speech act для каждого surface candidate;
- заменить exact-phrase тесты там, где текст должен стать вариативным, на семантические
  проверки.

### Этап 4. Active selection без обучения

- включить новый winner для реактивных direct mentions;
- оставить initiative на отдельном scheduling gate;
- neutral fallback создавать только после порога;
- stable cache сделать кандидатом или versioned decision cache;
- сохранить feature weights неизменными до накопления статистики.

### Этап 5. Единая атрибуция фидбека

- создать `FeedbackAttributor`;
- перевести ExperienceConsolidator и SelfDatasetRepository на общий сигнал;
- записывать arbitration feedback идемпотентно;
- расширить reflection strategies source/renderer признаками только там, где это не дублирует
  learner.

### Этап 6. Ограниченное обучение весов

- добавить versioned profiles;
- реализовать pairwise bounded update;
- начать с guild-level профиля;
- добавить rollback, baseline decay и admin diagnostics;
- включать только отдельным feature flag.

### Этап 7. Удаление legacy-приоритетов

- после подтверждённой стабильности удалить ранние selection returns;
- оставить hard validation, transport failures и provider applicability;
- обновить README и команды диагностики;
- сохранить legacy adapter минимум на один релиз для быстрого отката.

## 18. Тестирование

### 18.1. Unit tests

- каждый provider возвращает proposal или пустой tuple без выбора winner;
- порядок provider-ов не меняет winner;
- одинаковые вход, seed и weight version дают одинаковое решение;
- признаки находятся в допустимом диапазоне;
- missing required fact делает candidate ineligible;
- низкая topic relevance уменьшает score, но сама по себе не создаёт replacement-текст;
- fallback не участвует, пока есть candidate выше порога;
- dedup объединяет evidence одинаковых поверхностей;
- source error не отменяет остальных;
- cache не создаёт ранний winner;
- weight update идемпотентен;
- слабое продолжение разговора не обновляет arbitration weights;
- short-act rejection обновляет профиль в сторону альтернатив;
- знак positive/penalty weight не может перевернуться;
- rollback восстанавливает предыдущую версию.

### 18.2. Integration tests

- для одного сообщения сохраняются все кандидаты и ровно один winner;
- реально отправленный message ID связывается с decision ID;
- последующий reply/reaction связывается с тем же решением;
- shadow mode не меняет отправляемый legacy text;
- active mode выбирает по score, а не по generation order;
- identity required facts сохраняются во всех допустимых перефразированиях;
- lexical retrieval и graph одновременно участвуют в арбитраже;
- short act и обычная dialogue ветка могут конкурировать при неоднозначности;
- context relevance поступает в score каждого кандидата;
- initiative остаётся deferred в активном разговоре;
- candidate retention удаляет старые поверхности, не удаляя агрегированные веса и feedback.

### 18.3. Регрессионные сценарии

- `ты ведь умный?` не превращается в сообщение о непонимании;
- `мордобой ты` не запускает определение слова как единственную ветку;
- `как меня зовут?` сохраняет фактический Discord display name;
- чужое упоминание остаётся смысловой сущностью;
- точное определение не проигрывает несвязанному красивому тексту;
- неизвестное слово сначала проходит доступные автономные источники;
- наружу не выходят `candidate score`, `внутренний разбор`, queue state и rejection reasons;
- нерелевантный retrieval dump проигрывает краткому связанному плану;
- разные допустимые формулировки одного плана сохраняют отрицание и адресата.

### 18.4. Изменение существующих тестов

Точные сравнения строк сохраняются для:

- фактических инвариантов;
- сериализации;
- migration/repository поведения;
- аварийных transport сообщений, если их текст является контрактом.

Для разговорных ответов тесты должны проверять:

- speech act;
- required facts;
- отсутствие запрещённых утечек;
- тему;
- адресата;
- уверенность;
- допустимый диапазон длины;
- присутствие нескольких различных surface candidates.

## 19. Критерии приёмки

Работа считается завершённой, когда одновременно выполнено следующее:

1. В active path ни один обычный смысловой источник не побеждает только из-за порядка `if`.
2. Для применимых источников собирается полный ограниченный набор предложений до выбора.
3. Content и surface arbitration разделены.
4. `MemoryPhraseGenerator` возвращает кандидаты, а не только выбранную строку.
5. Контекстный гейт не заменяет реактивный ответ готовым текстом.
6. Fallback создаётся только после отсутствия кандидата выше порога.
7. Все кандидаты, признаки, компоненты score и winner сохраняются.
8. `candidate_count` означает фактическое число кандидатов.
9. Фидбек однозначно связан с decision и обрабатывается единым атрибутором.
10. Обучение весов ограничено, версионировано, идемпотентно и отключаемо.
11. Наружу не выходит внутренняя диагностика.
12. Существующие и новые тесты проходят; shadow/active имеют отдельные integration tests.
13. При одинаковом входе, seed и весах winner воспроизводим.
14. Есть feature flag и быстрый откат на legacy path.

## 20. Что не требуется в этой работе

- подключение LLM или внешнего генеративного API;
- удаление всех `if` из языкового кода;
- вероятностный выбор заведомо неверного кандидата;
- обучение на каждом сообщении без атрибуции;
- публикация внутренних рассуждений;
- автоматическое включение exploration;
- переписывание словаря, морфологии, event pipeline или memory layer с нуля.

## 21. Рекомендуемая первая рабочая итерация

Первый implementation slice должен включать только этапы 1–2 в `shadow`-режиме:

- модели;
- migration/repository;
- общий feature extractor;
- providers-адаптеры над текущими источниками;
- новый context relevance measurement;
- арбитраж со статическими baseline weights;
- полное логирование;
- сравнение с legacy winner;
- unit и integration tests.

Это даст реальные данные о распределении кандидатов и масштабе признаков, не меняя ответы в
Discord. Перенос готовых разговорных фраз в свободные content plans и обучение весов следует
делать после того, как верхнеуровневый арбитраж станет наблюдаемым и воспроизводимым.
