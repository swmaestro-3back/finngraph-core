"""
gazetteer(사전) 기반 엔티티 추출기.

KPF-BERT NER 모델이 놓치거나 표기가 흔들리는 도메인 특화 개체명(국내 상장사, 미국 기업,
제품, 국가)을 사전과 flashtext로 정확히 잡아 canonical 표준명으로 정규화한다.
NER 모델과 함께(동시에) 실행되어 결과가 병합된다 (workflow.ner_node 참고).
"""
from flashtext import KeywordProcessor

from app.graph.models import Entity, EntityLabel
from app.graph.ontology.gazetteers.country_dict import country_dict
from app.graph.ontology.gazetteers.krx_dict import krx_company_dict
from app.graph.ontology.gazetteers.product_dict import PRODUCT_DICT
from app.graph.ontology.gazetteers.us_dict import us_company_dict

# 각 EntityLabel에 매핑되는 사전들. krx(국내 상장사)와 us(미국 상장사)는 둘 다 COMPANY다.
# 사전은 {canonical: [표면형, ...]} 형태라 flashtext.add_keywords_from_dict에 그대로 넣을 수 있다.
GAZETTEERS: dict[EntityLabel, list[dict[str, list[str]]]] = {
    "COMPANY": [krx_company_dict, us_company_dict],
    "COUNTRY": [country_dict],
    "PRODUCT": [PRODUCT_DICT],
}

class Gazetteer:
    def __init__(self):
        # 라벨별 KeywordProcessor를 미리 구축해둔다. 사전에 표면형별 대/소문자 변형이
        # 이미 명시돼 있어 case_sensitive=True로 두어 과매칭('PR'이 'pr'에 걸리는 등)을 막는다.
        self._processors: dict[EntityLabel, KeywordProcessor] = {}
        # 정규화(치환) 전용 통합 프로세서. 라벨과 무관하게 모든 사전의 별칭을 한 번의
        # replace_keywords 호출로 canonical 표준명으로 바꾸기 위해 별도로 하나 더 둔다.
        self._normalizer = KeywordProcessor(case_sensitive=True)
        for label, dicts in GAZETTEERS.items():
            processor = KeywordProcessor(case_sensitive=True)
            for gazetteer in dicts:
                # add_keywords_from_dict: {clean_name: [keywords]} → 매칭 시 clean_name(canonical)을 반환
                processor.add_keywords_from_dict(gazetteer)
                self._normalizer.add_keywords_from_dict(gazetteer)
            self._processors[label] = processor

    def normalize(self, text: str) -> str:
        """본문의 모든 사전 별칭(표면형)을 canonical 표준명으로 치환한 새 텍스트를 반환한다.

        chunking 이전에 호출해 이후 NER/SRL이 흔들리는 표기 대신 일관된 표준명만 보도록 한다
        (예: '삼전'→'삼성전자', 'LG엔솔'→'LG에너지솔루션'). SRL 그라운딩이 원문 표면형과
        NER 엔티티 텍스트의 정확 일치에 의존하므로, 원문 자체를 표준화하는 것이 핵심이다.
        """
        return self._normalizer.replace_keywords(text)

    def extract_entities(self, text: str) -> list[Entity]:
        """본문에서 사전에 등록된 개체명을 찾아 canonical 표준명 Entity로 반환한다."""
        entities: list[Entity] = []
        for label, processor in self._processors.items():
            # extract_keywords는 본문 표면형이 아니라 매칭된 canonical 표준명 리스트를 반환한다.
            for canonical in processor.extract_keywords(text):
                entities.append(Entity(text=canonical, label=label))
        return entities
