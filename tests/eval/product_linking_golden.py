"""Measure item -> category accuracy against the real taxonomy and LLM.

Run: uv run python tests/eval/product_linking_golden.py

What it reports:
  1. Alias hit rate — the share resolved without an LLM call. This is the number that should
     climb over time as learned_aliases accumulate.
  2. Accuracy on the LLM-resolved remainder, with every miss printed as
     (item_text, expected, got) so a wrong call points straight at the definition to fix.
  3. Null rate — items the model declined to classify. A high rate means the taxonomy has a
     genuine gap, not that the prompt is broken.
"""

import asyncio
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.graph.nodes.product_linker import (
    ProductLinker,
    build_alias_processor,
    match_category,
)
from app.graph.ontology.product_taxonomy import load_taxonomy

# (item_text, source_sentence, expected category id)
# Hand-labelled following each category's definition in product_taxonomy.json.
_GOLDEN: list[tuple[str, str, str]] = [
    ("HBM3E 12단", "SK하이닉스는 엔비디아에 HBM3E 12단을 공급한다.", "메모리 반도체"),
    ("고대역폭메모리", "삼성전자는 고대역폭메모리를 공급하기로 했다.", "메모리 반도체"),
    ("서버용 D램", "삼성전자는 서버용 D램을 공급한다.", "메모리 반도체"),
    ("차량용 5나노 AI 칩", "삼성전자는 테슬라에 차량용 5나노 AI 칩을 공급한다.", "시스템 반도체"),
    ("전력반도체 모듈", "DB하이텍은 전력반도체 모듈을 공급한다.", "시스템 반도체"),
    ("TC본더", "한미반도체는 SK하이닉스에 TC본더를 공급한다.", "반도체 장비"),
    ("하이브리드 본딩 장비", "한미반도체는 하이브리드 본딩 장비를 공급한다.", "반도체 장비"),
    ("EUV용 블랭크 마스크", "에스앤에스텍은 EUV용 블랭크 마스크를 공급한다.", "반도체 부품/소재"),
    ("극자외선 포토레지스트", "동진쎄미켐은 극자외선 포토레지스트를 공급한다.", "반도체 부품/소재"),
    ("FC-BGA 기판", "삼성전기는 FC-BGA 기판을 공급한다.", "반도체 기판"),
    ("유리기판", "SKC는 유리기판을 공급하기로 했다.", "반도체 기판"),
    ("12인치 실리콘 웨이퍼", "SK실트론은 12인치 실리콘 웨이퍼를 공급한다.", "반도체 웨이퍼"),
    ("SiC 웨이퍼", "SK실트론은 SiC 웨이퍼를 공급한다.", "반도체 웨이퍼"),
    ("파운드리 위탁생산 물량", "삼성전자는 테슬라의 파운드리 위탁생산 물량을 수주했다.", "파운드리"),
    ("하이니켈 양극재", "에코프로비엠은 삼성SDI에 하이니켈 양극재를 공급한다.", "양극재"),
    ("단결정 NCM 양극재", "포스코퓨처엠은 단결정 NCM 양극재를 공급한다.", "양극재"),
    ("실리콘 음극재", "대주전자재료는 실리콘 음극재를 공급한다.", "음극재"),
    ("전고체 배터리 셀", "삼성SDI는 전고체 배터리 셀을 공급하기로 했다.", "배터리 셀"),
    ("46파이 원통형 배터리", "LG에너지솔루션은 46파이 원통형 배터리를 공급한다.", "배터리 셀"),
    ("리튬염 첨가제", "엔켐은 리튬염 첨가제를 공급한다.", "전해액"),
    ("습식 분리막", "SK아이이테크놀로지는 습식 분리막을 공급한다.", "분리막"),
    ("전지박", "SK넥실리스는 전지박을 공급한다.", "2차전지 장비/부품"),
    ("초고압 변압기", "효성중공업은 초고압 변압기를 공급한다.", "전력·그리드 기기"),
    ("액침냉각 시스템", "GS칼텍스는 액침냉각 시스템을 공급한다.", "데이터센터 인프라"),
    ("8세대 OLED 증착기", "선익시스템은 8세대 OLED 증착기를 공급한다.", "디스플레이"),
    ("LNG운반선 2척", "HD현대중공업은 LNG운반선 2척을 수주했다.", "조선·해양"),
    ("파라자일렌", "에쓰오일은 파라자일렌을 공급한다.", "석유화학 제품"),
    ("폴더블 스마트폰", "삼성전자는 폴더블 스마트폰을 공급한다.", "IT·가전 기기"),
    ("초소형 MLCC", "삼성전기는 애플에 초소형 MLCC를 공급한다.", "전자부품/첨단소재"),
    ("황산니켈", "고려아연은 황산니켈을 공급한다.", "비철금속"),
    ("열연강판", "포스코는 열연강판을 공급한다.", "철강"),
]


async def main() -> None:
    categories = load_taxonomy()
    processor = build_alias_processor(categories)

    alias_hits: list[tuple[str, str, str]] = []
    needs_llm: list[tuple[str, str, str]] = []
    for item_text, sentence, expected in _GOLDEN:
        got = match_category(processor, item_text)
        if got is None:
            needs_llm.append((item_text, sentence, expected))
        else:
            alias_hits.append((item_text, expected, got))

    print(f"total: {len(_GOLDEN)}")
    print(f"alias hit rate: {len(alias_hits)}/{len(_GOLDEN)}")

    alias_wrong = [row for row in alias_hits if row[1] != row[2]]
    print(f"alias accuracy: {len(alias_hits) - len(alias_wrong)}/{len(alias_hits)}")
    for item_text, expected, got in alias_wrong:
        print(f"  ALIAS MISS {item_text!r}: expected {expected}, got {got}")

    if not needs_llm:
        return

    linker = ProductLinker()
    result = await linker._chain.ainvoke(
        {
            "categories": "\n".join(
                f"- {c['id']} ({c['kind']}): {c['definition']}" for c in categories
            ),
            "items": "\n".join(
                f"- item_text: {item}\n  source_sentence: {sentence}"
                for item, sentence, _ in needs_llm
            ),
        }
    )
    got_by_item = {entry.item_text: entry.category_id for entry in result.items}

    outcomes: Counter[str] = Counter()
    for item_text, _, expected in needs_llm:
        got = got_by_item.get(item_text)
        if got is None:
            outcomes["null"] += 1
            print(f"  NULL {item_text!r}: expected {expected}")
        elif got == expected:
            outcomes["correct"] += 1
        else:
            outcomes["wrong"] += 1
            print(f"  LLM MISS {item_text!r}: expected {expected}, got {got}")

    print(
        f"llm accuracy: {outcomes['correct']}/{len(needs_llm)} "
        f"(wrong {outcomes['wrong']}, null {outcomes['null']})"
    )


if __name__ == "__main__":
    asyncio.run(main())
