"""
나중에 할 것

1. kiwipipey의 분석 결과를 전체 다 들고 가지말고 필요한 것들만 추출해서 State로 넘기기
2. 필요없는거 None 처리하거나 Subgraph 처리하기
"""

from kiwipiepy import Kiwi, Sentence

_kiwi = Kiwi()

def _extract_verb_lemmas(tokens: list) -> list[str]:
    """동사 원형을 추출한다.

    한국어 동사의 kiwipiepy 분석 패턴:
      - 순수 동사: VV/VV-R/VV-I → tok.lemma  (불규칙 포함: 돕다, 묻다)
      - 명사+하다 복합동사: NNG/NNP + (조사*) + XSV → NNG.form + XSV.form + "다"
        (예: 인수(NNG) + 하(XSV) → 인수하다)
        (예: 인수(NNG) + 를(JKO) + 하(XSV) → 인수하다)
    """
    lemmas: list[str] = []

    for i, tok in enumerate(tokens):
        tag = str(tok.tag)

        if tag.startswith("VV"):
            # startswith로 VV-R(규칙), VV-I(불규칙) 변형 태그를 모두 포함
            # tok.lemma 사용: form + "다" 방식은 불규칙 동사에서 틀림 (도와 → 도와다 X, 돕다 O)
            lemmas.append(tok.lemma)

        elif tag.startswith("XSV"):
            # XSV는 "하다" 계열 동사 파생 접미사 (예: 인수하다의 "하")
            # 복합동사 원형을 만들기 위해 직전 NNG/NNP를 역방향으로 탐색
            # 조사(J*)가 NNG와 XSV 사이에 끼는 경우를 처리: 인수(NNG) + 를(JKO) + 하(XSV)
            # 모든 조사 태그는 세종 태그셋에서 J로 시작하므로 startswith("J")로 일괄 스킵
            nng_form = None
            for j in range(i - 1, -1, -1):
                prev_tag = str(tokens[j].tag)
                if prev_tag.startswith("J"):
                    continue
                if prev_tag in ("NNG", "NNP"):
                    nng_form = tokens[j].form
                break  # 조사도 NNG/NNP도 아닌 토큰을 만나면 탐색 중단

            if nng_form:
                lemmas.append(nng_form + tok.form + "다")  # 인수 + 하 + 다 → 인수하다
            else:
                lemmas.append(tok.form + "다")  # 문맥 없는 XSV → 하다

    return list(dict.fromkeys(lemmas))  # 중복 제거, 순서 유지


class MorphAnalyzer:
    def split_sentences(self, text: str) -> list[Sentence]:
        """kiwi 기준으로 문장 분리와 형태소 분석을 함께 수행한다.

        문장 분리는 내부적으로 형태소 분석에 기반하므로(kiwipiepy 명세), chunking_node가
        이 결과의 문장 표면형은 NER로, analyze()는 같은 결과의 tokens를 받아 재사용해
        형태소 분석이 중복 수행되지 않도록 한다.
        """
        return _kiwi.split_into_sents(text, return_tokens=True)

    def analyze(self, sentences: list[Sentence]) -> list[str]:
        """문서 전체에서 등장하는 동사 원형을 중복 제거해 반환한다 (SRL 프롬프트 힌트용)."""
        all_lemmas: list[str] = []
        for sent in sentences:
            all_lemmas.extend(_extract_verb_lemmas(sent.tokens))

        return list(dict.fromkeys(all_lemmas))
