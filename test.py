import os

from app.graph.workflow import GraphRunner
from app.core.config import settings

if settings.LANGSMITH_TRACING:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT

TEXT="""
[효성重, 호주서 3100억 초고압변압기 수주..5년 독점 공급]

효성중공업은 지난 1일(현지시간) 호주 빅토리아주의 유일한 송전망 운영사이자 최대 에너지 네트워크 기업인 오스넷(AusNet)과 초고압변압기, 리액터 등 전력기기 장기공급계약을 체결했다고 2일 밝혔다.

예상 총 수주액은 약 3100억원 규모다. 이 계약으로 효성중공업은 향후 5년간 호주 빅토리아주 송전망에 초고압 전력기기를 독점 공급하게 된다.

이는 지난 3월 호주 퀸즐랜드주에서 수주한 1425억원 규모 에너지저장장치(ESS) 프로젝트에 연이은 대규모 수주다.

이번 계약으로 효성중공업은 빅토리아주를 비롯해 퀸즐랜드, 뉴사우스웨일스, 남호주 등 호주 주요 지역에 초고압 전력기기를 공급하며 호주 전역을 아우르는 공급업체로서 입지를 더욱 강화하게 됐다. 현재 효성중공업은 호주 송전시장 초고압변압기 부문 점유율 1위를 기록하고 있다.

효성중공업은 이번 오스넷과의 계약이 초고압변압기 공급을 넘어 향후 초고압직류송전(HVDC), 전력계통 안정화 장치인 정지형 무효전력 보상장치(STATCOM) 등 차세대 전력망 솔루션 분야로 협력을 확대할 수 있는 기반을 마련했다는 점에서도 의미가 크다고 설명했다.

호주 정부는 가파른 신재생에너지 전환에 따른 전력망 불안정성 해소와 대규모 장거리 송전망 확충을 위해 200억 호주달러(약 20조원) 규모의 '국가 전력망 재정비(Rewiring the Nation)' 사업을 본격 추진하고 있다. 이에 따라 빅토리아, 뉴사우스웨일스 등을 연결하는 주간 송전망 연계 프로젝트와 각 지역의 핵심 신재생에너지 구역 내 전력 인프라 구축에 집중적인 투자가 이뤄지고 있다.

조현준 효성그룹 회장은 "호주는 에너지 전환의 속도와 규모 면에서 세계에서 가장 역동적인 시장 중 하나"라며 "앞으로도 HVDC, STATCOM 등 차세대 전력망 솔루션까지 협력을 확대하며 호주 에너지 전환을 함께 이끄는 파트너가 되겠다"고 말했다.
"""


def main() -> None:
    runner = GraphRunner()
    runner.invoke("1", TEXT)
    print("Graph completed successfully.")
    
if __name__ == "__main__":
    main()
