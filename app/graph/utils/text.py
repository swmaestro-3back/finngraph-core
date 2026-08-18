import re


# LLM이 verbatim 지시를 어겨 공백·개행만 미묘하게 바꾸는 경우에도 원문 대조가 통과하도록,
# 비교 전에 모든 공백을 제거한다. RelationExtractor의 근거 문장 검증과 FrameAnnotator의
# evidence 그라운딩 검증이 같은 규칙을 써야 하므로 공용 모듈로 둔다.
def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", "", s)
