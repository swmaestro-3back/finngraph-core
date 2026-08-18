"""앱 진입점.

실행: python -m app.main

Neo4j에 데이터가 없으면 먼저 시드를 적재한 뒤, 그래프 워크플로우를 실행한다.
"""

import asyncio
import uuid

from langchain_core.tracers.langchain import wait_for_all_tracers

from app.crud import upsert_triplets
from app.scripts.seed_db import seed
from app.graph.workflow import GraphRunner

TEXT="""
한미반도체가 미국 우주기업 스페이스X에 대한 전략적 투자 소식을 발표하면서 장 초반 강세를 보이고 있다.

한국거래소에 따르면 12일 오전 10시 31분 기준 한미반도체는 전 거래일 대비 4만2000원(14.43%) 오른 33만3000원에 거래되고 있다.

한미반도체는 이날 일론 머스크가 이끄는 우주기업 스페이스X에 500억원 규모의 전략적 투자를 단행한다고 밝혔다. 회사 측은 인공지능(AI) 인프라 확대와 우주·위성통신 산업 성장에 선제적으로 대응하기 위한 차원이라고 설명했다.

이번 투자에는 한미반도체와 피터 틸 계열 투자 네트워크의 인연도 작용한 것으로 전해졌다. 한미반도체는 과거 반도체 장비업체 HPSP 투자 등을 통해 투자 수익을 거둔 바 있으며, 성장성이 높은 기업에 대한 전략적 투자를 이어가고 있다.

한미반도체는 향후 투자 성과를 반도체 장비 사업에 재투자해 성장 동력을 강화하고 기업가치 제고에 나설 계획이다.

한미반도체(042700)가 SK하이닉스와 HBM4 제조 장비 공급계약을 체결했다.

◆ SK하이닉스에 HBM4 제조용 'TC BONDER 4.5 GRIFFIN' 공급

한미반도체는 SK하이닉스와 HBM4 제조용 'TC BONDER 4.5 GRIFFIN' 장비 수주 계약을 체결했다고 8일 공시했다. 계약(수주)일자는 발주서(P/O)를 수령한 2026년 6월 8일이다.

이번 계약 금액은 총 442억 원(VAT 별도) 규모다. 이는 한미반도체의 2025년 연결기준 최근 매출액인 5,766억 8,461만 원의 7.66%에 해당하는 수준이며, 계약기간은 2026년 9월 2일까지다. 계약금 선급금은 없으며, 계약종료일(납기)은 고객사와의 협의에 따라 변경될 수 있다.

한미반도체는 1980년 반도체 자동화 장비의 제조 및 판매를 목적으로 설립되었으며, 2005년 유가증권시장에 상장했다. 주물, 설계, 제작, 조립, 검사와 테스트까지의 수직통합제조 시스템을 구축하여 글로벌 주요 AI 반도체 기업에 최첨단 장비를 공급하고 있다.

주력 장비인 HBM TC 본더는 메모리칩에 정밀한 열과 압력을 가해 고적층 HBM을 생산하는 핵심 장비다. 아울러 MSVP는 세계 시장점유율 1위를 유지하고 있다.
"""

async def main() -> None:
    # 실행 전에 DB가 비어있으면 시드부터 채운다.
    await seed()

    runner = GraphRunner()
    try:
        news_id = str(uuid.uuid4())
        final_state = await runner.ainvoke(news_id, TEXT)

        # GraphDB에 반영
        # await upsert_triplets(news_id, final_state["triplets"])

        print(
            f"Graph completed successfully. "
            f"{len(final_state['triplets'])} triplets extracted."
        )
    finally:
        # LangSmith SDK는 트레이스를 백그라운드 스레드에서 배치 업로드한다.
        # 짧게 실행되는 스크립트는 root(전체 workflow) run의 종료·output 이벤트가
        # 올라가기 전에 프로세스가 끝나버려 LangSmith에서 최상위 output이 비어 보일 수 있다.
        # 종료 전에 모든 트레이서가 flush를 마치도록 대기한다.
        wait_for_all_tracers()


if __name__ == "__main__":
    asyncio.run(main())
