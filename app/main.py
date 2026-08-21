import asyncio
import uuid

from langchain_core.tracers.langchain import wait_for_all_tracers

from app.crud import upsert_triplets
from app.scripts.seed_db import seed
from app.graph.workflow import GraphRunner

TEXT = """
엔비디아가 인공지능(AI) 추론에 최적화한 그록(Groq)3 언어처리장치(LPU)와 이를 탑재한 그록3 LPX 랙을 출시한다. 엔비디아 차세대 AI 플랫폼 베라 루빈과 연결해 사용할 수 있다. LPU 칩 생산은 삼성전자가 맡았다.
젠슨 황 엔비디아 최고경영자(CEO)는 16일(현지시간) 캘리포니아 새너제이에서 열린 엔비디아 연례 개발자 행사(GTC) 2026 기조연설에서 그록3 LPU를 선보였다. 대형언어모델(LLM)에 최적화된 처리장치다. 엔비디아는 지난해 12월 AI 추론용 칩 설계회사(팹리스) 그록을 우회 인수하며 해당 기술의 라이선스 계약을 체결한 바 있다.
그록3는 D램 대비 속도가 빠른 S램 메모리를 탑재했다. 메모리 대역폭이 초당 22테라바이트(TB/s)에서 최대 150TB/s에 달한다. HBM보다 7~45배 빠른 수치다. AI 토큰 생성 지연 시간을 기존 HBM 대비 크게 줄일 수 있다. 칩 생산은 삼성전자 파운드리가 수행 중이다. 황 CEO는 "LPU 칩을 제조하고 있는 삼성전자에 감사의 말을 전한다"며 "삼성전자는 현재 최대한 칩을 찍어내고 있다"고 언급했다.
단점은 메모리 용량이다. S램은 D램 대비 생산 비용이 비싸고 칩 면적이 크다. 이 때문에 그록3는 약 500MB S램을 지닌다. 엔비디아 루빈 그래픽처리장치(GPU) 하나당 288GB 삼성전자 6세대 고대역폭메모리(HBM4)를 탑재한 것과 비교하면 500분의 1 수준이다. 이 때문에 그록3 LPU만으로 조 단위 매개변수(파라미터)를 지닌 초거대 AI 모델을 구동하려면 수천 개의 칩을 병렬 연결해야 한다. 이 때 칩당 토큰 처리량(경제성)은 크게 떨어진다.
실제로 엔비디아는 이 LPU를 256개 탑재한 그록3 LPX 랙을 만들었다. 대역폭은 640TB/s로 높으나 S램 용량은 128GB에 불과하다. 메모리 용량 한계가 뚜렷하다. 엔비디아는 이 단점을 극복하고자 LPX 랙을 베라 루빈 플랫폼과 결합하는 전략을 택했다. 그록3 LPX 랙의 높은 메모리 대역폭과 루빈 GPU의 높은 부동소수점 연산 성능(FLOPS), 대용량 HBM 장점을 합친 것이다.
엔비디아는 맞춤형 AI용 이더넷 연결 플랫폼 스펙트럼-X로 두 시스템을 연결했다. 이어 엔비디아가 새로 도입한 소프트웨어 다이나모(Dynamo)가 각 시스템별 역할을 자동 할당해 AI 모델 토큰을 빠르게 처리한다. 추론 과정에서 높은 대역폭이 필요한 순방향신경망(FFN) 계층 연산은 LPU로 넘긴다. FFN 연산은 각 토큰의 의미를 맥락에 맞게 처리하는 과정이다. 복잡한 수학 연산과 큰 키-값(밸류) 메모리가 필요한 어텐션과 AI 모델 나머지 부분은 GPU가 처리한다. 어텐션은 각 토큰 사이 관계를 분석하는 것이다. 두 시스템을 단일 컴퓨터처럼 묶은 엔드투엔드(E2E) 시스템 공동 설계다.
엔비디아는 두 랙의 결합으로 초거대 AI 모델 구동 시 100만 토큰당 45달러의 비용이 든다고 강조했다. 초당 토큰 처리량은 500이다. 이는 기존 전력당 추론 처리량(스루풋) 대비 35배 향상된 수치다. 엔비디아는 초거대 AI 모델 서비스 수익 창출 기회가 10배 증가한다고 주장한다. 기존 엔비디아 GPU 소프트웨어 플랫폼인 쿠다(CUDA) 생태계를 수정할 필요도 없다.
엔비디아 그록3 LPX 랙은 올해 하반기 베라 루빈 플랫폼 정식 출시 일정에 맞춰 시장에 공급될 예정이다. 본격적인 출하 시점은 올해 3분기다. 그 외 베라 루빈 시스템을 구성하는 베라 중앙처리장치(CPU)와 루빈 GPU, NV링크 6 스위치, 커넥트X-9 네트워크 인터페이스 카드(NIC), 블루필드-4 데이터처리장치(DPU), 스펙트럼-6 이더넷 스위치 칩들은 양산 단계에 돌입했다.
황 CEO는 "베라 루빈 초기 샘플 공급은 전 세대와 달리 성공적으로 이뤄졌다"며 "마이크로소프트 애저 데이터센터에 첫 랙이 설치돼 가동을 시작했다"고 밝혔다. 엔비디아는 아마존 웹 서비스, 구글 클라우드, 오라클 클라우드에도 베라 루빈을 공급할 예정이다. 오픈AI와 앤트로픽, 메타 등과도 협력해 AI 모델 최적화와 안정적인 서비스 환경 구축을 지원할 계획이다.
"""


async def main() -> None:
    # Ensure database is seeded before running
    await seed()

    runner = GraphRunner()
    try:
        # example news uuid
        news_id = str(uuid.uuid4())
        final_state = await runner.ainvoke(news_id, TEXT)

        # Upsert to neo4j
        await upsert_triplets(news_id, final_state["triplets"])

        print(
            f"Graph completed successfully. "
            f"{len(final_state['triplets'])} triplets extracted."
        )
    finally:
        # LangSmith SDK uploads traces asynchronously via background threads.
        # Short-lived scripts may terminate before root workflow run events (end/output)
        # are fully flushed, leaving the top-level trace output empty in LangSmith.
        # Ensure all pending traces are flushed before process exit.
        wait_for_all_tracers()


if __name__ == "__main__":
    asyncio.run(main())
