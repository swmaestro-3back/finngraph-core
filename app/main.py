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

TEXT = """
인공지능(AI) 반도체 기업 엔비디아가 네이버에 대규모 투자를 단행한다는 소식에 네이버 주가가 27일 장중 급등했다.

이날 한국거래소에 따르면 오전 10시 14분 현재 네이버(NAVER)는 전 거래일 대비 8.67% 오른 22만5,500원에 거래되고 있다. 장 초반에는 상승폭이 10%를 넘기도 했다.

주가를 끌어올린 것은 엔비디아의 전략적 투자 유치다. 네이버는 이날 엔비디아를 대상으로 약 1조4,809억원(10억달러) 규모의 제3자배정 유상증자를 결정했다고 공시했다. 발행가는 주당 20만4,500원으로, 엔비디아는 이번 투자로 네이버 지분 4.5%를 확보해 3대 주주에 오른다. 양사는 글로벌 AI 팩토리를 공동 구축하기로 했다. 네이버가 데이터센터 부지와 운영을 맡고 엔비디아가 그래픽처리장치(GPU)를 공급하는 구조다.

이날 증시에서는 AI 인프라 투자 기대감에 관련주가 동반 강세를 보였다. 네이버는 국내 대표 인터넷 검색·플랫폼 기업으로 커머스, 클라우드, AI 사업을 영위하고 있다.
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
        
        print(f"Graph completed successfully. {len(final_state['triplets'])} triplets saved.")
    finally:
        # LangSmith SDK는 트레이스를 백그라운드 스레드에서 배치 업로드한다.
        # 짧게 실행되는 스크립트는 root(전체 workflow) run의 종료·output 이벤트가
        # 올라가기 전에 프로세스가 끝나버려 LangSmith에서 최상위 output이 비어 보일 수 있다.
        # 종료 전에 모든 트레이서가 flush를 마치도록 대기한다.
        wait_for_all_tracers()


if __name__ == "__main__":
    asyncio.run(main())
