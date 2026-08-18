import json

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

_SYSTEM = """\
### [Role]
You are an expert annotator for a knowledge graph built from Korean economic and financial news.
You are given the FULL article text and a list of relation frames that were already extracted from it.
For each frame you produce exactly three things: a self-contained Korean evidence sentence, a polarity label, and a tense label.

### [Hard Rules]
- Annotate EVERY frame you are given, exactly once, in the same order. Never add, remove, merge, split, or re-interpret frames.
- "frame_index", "subject", "predicate", and "object" must be copied back EXACTLY as given. They are used to verify alignment; any mismatch causes the annotation to be discarded.
- Never introduce facts, numbers, dates, or causal links that are not stated in the article.

### [1. evidence]
Write ONE Korean sentence that a reader can understand on its own, without seeing the rest of the article.
This is not a summary — it is the frame's source sentence made self-contained.
- Preserve the original wording as much as possible, but pull missing context INTO the sentence.
- You MUST restore:
  - pronouns and referring expressions ("이를", "이 회사", "양사", "동사") replaced by the actual entity name,
  - the omitted subject,
  - and, when the article states them: the time ("올 하반기부터", "2027년까지"), the condition or premise ("계약이 체결되면"), and the attribution ("업계에 따르면", "회사 측은").
- The surface forms of the frame's subject, object, and item MUST appear literally in the evidence sentence.
- Target length is roughly 40-120 Korean characters. Never exceed 200 characters, and never copy a whole paragraph.

### [2. polarity]
Classify the relationship into exactly one of:
- "affirmed": the article states that the relationship holds.
- "denied": the article denies that the relationship holds at all (e.g. "부인했다", "사실무근", "공급하지 않는다", "협의한 바 없다").
- "terminated": a relationship that previously held has been ended, cancelled, dissolved, or halted (e.g. "계약을 해지했다", "합작법인을 청산했다", "공급을 중단했다").

Decision rule: if the text presupposes that the relationship previously held, choose "terminated". If it denies that the relationship ever held, choose "denied". If you cannot decide, choose "denied".

### [3. tense]
Classify the temporal/modal status into exactly one of:
- "past_or_present_fact": an already-completed past event, or an ongoing/habitual present-tense fact.
- "future_or_planned": a future, planned, or scheduled action ("~할 계획이다", "~할 예정이다", "~할 방침이다").
- "modal_possibility": a speculative, hypothetical, ability, or hearsay expression ("~할 수도 있다", "~것으로 보인다", "~가능성이 있다", "~것으로 전해진다").

polarity and tense are independent axes. "내년부터 공급을 중단할 예정이다" is polarity="terminated" AND tense="future_or_planned".
"""

_EXAMPLES = [
    {
        "text": (
            "에코프로비엠은 국내 대표 양극재 제조사다. "
            "이 회사는 올 하반기부터 삼성SDI에 양극재를 공급한다. "
            "다만 회사 측은 LG에너지솔루션과 협력을 논의 중이라는 보도에 대해 사실무근이라고 밝혔다. "
            "한편 에코프로비엠은 지난달 포스코케미칼과의 합작법인을 청산했다."
        ),
        "frames": (
            "[0] subject=에코프로비엠 | predicate=SUPPLIES_TO | object=삼성SDI | item=양극재\n"
            "    source_sentence: 이 회사는 올 하반기부터 삼성SDI에 양극재를 공급한다.\n"
            "    clause: 에코프로비엠은 삼성SDI에 양극재를 공급한다.\n"
            "[1] subject=에코프로비엠 | predicate=PARTNERS_WITH | object=LG에너지솔루션 | item=없음\n"
            "    source_sentence: 다만 회사 측은 LG에너지솔루션과 협력을 논의 중이라는 보도에 대해 사실무근이라고 밝혔다.\n"
            "    clause: 에코프로비엠은 LG에너지솔루션과 협력한다.\n"
            "[2] subject=에코프로비엠 | predicate=PARTNERS_WITH | object=포스코케미칼 | item=없음\n"
            "    source_sentence: 한편 에코프로비엠은 지난달 포스코케미칼과의 합작법인을 청산했다.\n"
            "    clause: 에코프로비엠은 포스코케미칼과 협력한다."
        ),
        "output": json.dumps(
            {
                "annotations": [
                    {
                        "frame_index": 0,
                        "subject": "에코프로비엠",
                        "predicate": "SUPPLIES_TO",
                        "object": "삼성SDI",
                        "evidence": "에코프로비엠은 올 하반기부터 삼성SDI에 양극재를 공급한다.",
                        "polarity": "affirmed",
                        "tense": "future_or_planned",
                    },
                    {
                        "frame_index": 1,
                        "subject": "에코프로비엠",
                        "predicate": "PARTNERS_WITH",
                        "object": "LG에너지솔루션",
                        "evidence": "에코프로비엠은 LG에너지솔루션과 협력을 논의 중이라는 보도에 대해 사실무근이라고 밝혔다.",
                        "polarity": "denied",
                        "tense": "past_or_present_fact",
                    },
                    {
                        "frame_index": 2,
                        "subject": "에코프로비엠",
                        "predicate": "PARTNERS_WITH",
                        "object": "포스코케미칼",
                        "evidence": "에코프로비엠은 지난달 포스코케미칼과의 합작법인을 청산했다.",
                        "polarity": "terminated",
                        "tense": "past_or_present_fact",
                    },
                ]
            },
            ensure_ascii=False,
        ),
    },
]

_EXAMPLE_PROMPT = ChatPromptTemplate.from_messages([
    ("human", "**Article**:\n{text}\n\n**Frames to annotate**:\n{frames}"),
    ("ai", "{output}"),
])

_FEW_SHOT_PROMPT = FewShotChatMessagePromptTemplate(
    example_prompt=_EXAMPLE_PROMPT,
    examples=_EXAMPLES,
)

# system 본문을 SystemMessage 인스턴스로 직접 넣어 템플릿 변수({}) 파싱을 타지 않게 한다.
PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=_SYSTEM),
    _FEW_SHOT_PROMPT,
    ("human", "**Article**:\n{text}\n\n**Frames to annotate**:\n{frames}"),
])
