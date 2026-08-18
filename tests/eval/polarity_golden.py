"""polarity 3분류 정확도와 evidence 자립성을 실제 LLM으로 측정하는 평가 스크립트.

실행: uv run python tests/eval/polarity_golden.py

측정 항목
  1. polarity 혼동 행렬 — 특히 denied ↔ terminated 혼동이 핵심 관심사다.
     둘을 가르는 규칙("이전 성립을 전제하면 terminated")이 프롬프트에서 실제로 작동하는지 본다.
  2. evidence 자립성 — FrameAnnotator의 그라운딩 검증을 통과한 비율과 길이 분포.
     드롭률이 높으면 evidence 규칙이 너무 빡빡하거나 프롬프트가 지켜지지 않는다는 신호다.
"""

import asyncio
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.graph.nodes.entity_extractor import EntityExtractor
from app.graph.nodes.frame_annotator import FrameAnnotator
from app.graph.nodes.relation_extractor import RelationExtractor

# (기사 본문, {(subject, predicate, object): 기대 polarity})
# 기대값은 스펙 §6의 결정 규칙에 따라 수동 라벨링했다.
#
# 참고: EntityExtractor.canonicalize()는 gazetteer에 등록된 이표기(surface form)를
# 표준 명칭(canonical name)으로 치환한다. 아래 "포스코케미칼" 관련 사례는
# gazetteer 상 표준 명칭이 "포스코퓨처엠"이라 정규화 후 엔티티 텍스트가
# "포스코퓨처엠"이 된다. 따라서 기대 관계 키(expectations)는 기사 문구가 아니라
# 정규화된 표준 명칭 "포스코퓨처엠"을 사용한다. 문장 자체는 원문 그대로 유지한다
# (normalize()가 자동으로 치환하므로 문법적 형태·의미는 변하지 않는다).
_GOLDEN: list[tuple[str, dict[tuple[str, str, str], str]]] = [
    (
        "에코프로비엠은 올 하반기부터 삼성SDI에 양극재를 공급한다.",
        {("에코프로비엠", "SUPPLIES_TO", "삼성SDI"): "affirmed"},
    ),
    (
        "에코프로비엠은 LG에너지솔루션과 협력을 논의 중이라는 보도에 대해 사실무근이라고 밝혔다.",
        {("에코프로비엠", "PARTNERS_WITH", "LG에너지솔루션"): "denied"},
    ),
    (
        "에코프로비엠은 지난달 포스코케미칼과의 합작법인을 청산했다.",
        {("에코프로비엠", "PARTNERS_WITH", "포스코퓨처엠"): "terminated"},
    ),
    (
        "삼성SDI는 코스모신소재로부터 양극재를 공급받아 왔으나 지난달 공급 계약을 해지했다.",
        {("코스모신소재", "SUPPLIES_TO", "삼성SDI"): "terminated"},
    ),
    (
        "삼성전자는 인텔 인수설에 대해 검토한 바 없다고 선을 그었다.",
        {("삼성전자", "ACQUIRES", "인텔"): "denied"},
    ),
    (
        "코스모신소재는 내년부터 삼성SDI에 대한 양극재 공급을 중단할 예정이다.",
        {("코스모신소재", "SUPPLIES_TO", "삼성SDI"): "terminated"},
    ),
]


async def _run_article(
    entity_extractor: EntityExtractor,
    relation_extractor: RelationExtractor,
    frame_annotator: FrameAnnotator,
    article: str,
):
    normalized = entity_extractor.canonicalize(article)
    entities = entity_extractor.extract(normalized)
    candidates = await relation_extractor.extract(normalized, entities)
    frames, stats = await frame_annotator.annotate(normalized, candidates)
    return candidates, frames, stats


async def main() -> None:
    entity_extractor = EntityExtractor()
    relation_extractor = RelationExtractor()
    frame_annotator = FrameAnnotator()

    # confusion[기대값][실제값] = 개수
    confusion: dict[str, Counter] = defaultdict(Counter)
    not_extracted = 0
    total_candidates = 0
    total_annotated = 0
    dropped_mismatch = 0
    dropped_grounding = 0
    evidence_lengths: list[int] = []
    records: list[dict] = []

    for article, expectations in _GOLDEN:
        candidates, frames, stats = await _run_article(
            entity_extractor, relation_extractor, frame_annotator, article
        )
        total_candidates += len(candidates)
        total_annotated += len(frames)
        dropped_mismatch += stats["dropped_annotation_mismatch"]
        dropped_grounding += stats["dropped_evidence_grounding"]

        by_key = {
            (f.subject.text, f.predicate, f.object.text): f for f in frames
        }
        for key, expected in expectations.items():
            frame = by_key.get(key)
            if frame is None:
                not_extracted += 1
                records.append({"article": article, "key": list(key), "expected": expected, "actual": None})
                continue
            confusion[expected][frame.polarity] += 1
            evidence_lengths.append(len(frame.evidence))
            records.append(
                {
                    "article": article,
                    "key": list(key),
                    "expected": expected,
                    "actual": frame.polarity,
                    "tense": frame.tense,
                    "evidence": frame.evidence,
                    "source_sentence": frame.source_sentence,
                }
            )

    labels = ["affirmed", "denied", "terminated"]
    matched = sum(confusion[label][label] for label in labels)
    scored = sum(sum(confusion[label].values()) for label in labels)

    print("=== polarity 혼동 행렬 (행=기대, 열=실제) ===")
    print(f"{'':<12}" + "".join(f"{label:<13}" for label in labels))
    for expected in labels:
        row = "".join(f"{confusion[expected][actual]:<13}" for actual in labels)
        print(f"{expected:<12}{row}")

    print()
    print(f"polarity 정확도: {matched}/{scored}" + (f" ({matched / scored:.1%})" if scored else ""))
    print(f"골든 관계 미추출: {not_extracted}건")
    print()
    print("=== evidence 자립성 ===")
    print(f"후보 프레임: {total_candidates} → 주석 완료: {total_annotated}")
    print(f"에코 불일치·누락 드롭: {dropped_mismatch}")
    print(f"evidence 그라운딩 드롭: {dropped_grounding}")
    if evidence_lengths:
        evidence_lengths.sort()
        median = evidence_lengths[len(evidence_lengths) // 2]
        print(
            f"evidence 길이 min/median/max: "
            f"{evidence_lengths[0]}/{median}/{evidence_lengths[-1]}자"
        )

    output_path = Path(__file__).parent / "polarity_golden_result.json"
    output_path.write_text(
        json.dumps(
            {
                "confusion": {e: dict(c) for e, c in confusion.items()},
                "not_extracted": not_extracted,
                "total_candidates": total_candidates,
                "total_annotated": total_annotated,
                "dropped_annotation_mismatch": dropped_mismatch,
                "dropped_evidence_grounding": dropped_grounding,
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n결과 저장: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
