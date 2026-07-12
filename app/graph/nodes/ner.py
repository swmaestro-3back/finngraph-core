"""
현재 KPF-bert-ner 모델은 내부적으로 한 문장씩 청킹하여 처리한다.
따라서 문단을 넣어도 내부적으로 split sentences를 통해
문장 분할을 하고 for sent in sentences로 NER 작업을 진행한다.
따라서 현 프로젝트에서는 형태소 분할을 하는 kiwi도 한 문장 단위로 진행되어 split sentences를 하는 점을 파악하여
형태소분할할때 진행된 분할된 문장들을 그대로 사용하여 현재 NER 함수에는 하나의 문장이 입력으로 들어온다.
"""
import torch
from transformers import AutoTokenizer, BertForTokenClassification, logging as hf_logging

from app.graph.utils.kpf_labels import ID2LABEL, kpf_to_pipeline
from app.graph.models import Entity

#  transformers 라이브러리 자체의 로그 레벨을 ERROR로 올려서,
# 평소 INFO/WARNING 레벨로 출력되는 로딩 관련 메시지들을 조용히 시키는 용도
hf_logging.set_verbosity_error()

MODEL_PATH = "./KPF-bert-ner"
# chunking 노드가 이미 512자 이내로 잘라서 넘겨주므로 문장 단위 재분할 없이 청크를
# 통째로 추론한다. 모델의 max_position_embeddings가 512라 truncation으로 안전장치를 둔다.
MAX_LENGTH = 512

class NER:
    def __init__(self):
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self._model = BertForTokenClassification.from_pretrained(MODEL_PATH)
        self._model.eval()

    def extract_entities(self, text: str) -> list[Entity]:
        kpf_entries = self._ner_predict(text)

        seen: set[tuple[str, str]] = set()
        entities: list[Entity] = []

        for entry in kpf_entries:
            pipeline_label = entry["pipeline_label"]
            # pipeline_label=None: 파이프라인 태그셋(기업 간 관계)에 대응 없는 KPF 카테고리 → 제외
            if pipeline_label is None:
                continue
            key = (entry["word"], pipeline_label)
            if key in seen:
                continue
            seen.add(key)
            entities.append(Entity(text=entry["word"], label=pipeline_label))

        return entities

    # 여기서의 text는 cleaning node에 의해 줄바꿈과 같은 기호들이 모두 제거된 한 개의 문장.
    def _ner_predict(self, text: str) -> list[dict]:
        """KPF 공식 ner_module.py 로직 기반 (CPU 버전)."""
        # KPF 모델 학습 시 공백을 '-'로 치환했으므로 추론 시에도 동일하게 적용
        text_dashed = text.replace(" ", "-")
        inputs = self._tokenizer(
            text_dashed, return_tensors="pt", truncation=True, max_length=MAX_LENGTH
        )

        with torch.no_grad():
            outputs = self._model(**inputs)

        token_prediction_list = outputs.logits.argmax(dim=2).squeeze(0).tolist()
        pred_str = [ID2LABEL[l] for l in token_prediction_list]
        tt_tokens = self._tokenizer(
            text_dashed, truncation=True, max_length=MAX_LENGTH
        ).encodings[0].tokens

        word_list: list[dict] = []

        # 아래 루프는 토큰(서브워드) 단위 BIO 예측을 엔티티 단위 텍스트로 모으는
        # 상태 머신이다. 모델은 "삼성전자" 같은 한 단어도 서브워드로 쪼개어
        # B-OGG_ECONOMY / I-OGG_ECONOMY / I-OGG_ECONOMY 식으로 토큰마다 예측하므로,
        # 이어붙여서 하나의 엔티티로 복원하는 과정이 반드시 필요하다.
        is_prev_entity = False
        prev_entity_tag = ""
        # 모델이 B- 없이 I-로 엔티티를 시작하는(BIO 규칙 위반) 경우를 방어한다.
        # 이런 경우 _word는 계속 누적되지만 is_prev_entity=False로 유지되어,
        # 다음 O/B- 시점에 엔티티로 저장되지 않고 조용히 버려진다.
        is_there_B_before_I = False
        _word = ""

        for i, (token, pred) in enumerate(zip(tt_tokens, pred_str)):
            # i=0: [CLS], i=마지막: [SEP] — 특수 토큰이므로 건너뜀
            if i == 0 or i == len(pred_str) - 1:
                continue

            # '##' 제거로 서브워드 복원, '-'는 원래 공백이었으므로 되돌림
            token = token.replace('#', '').replace("-", " ")
            if token == "":
                continue

            if 'B-' in pred:
                # 새 엔티티의 시작. 직전에 다른 엔티티를 누적 중이었다면
                # (연속된 두 엔티티 사이에 O가 안 끼는 경우) 먼저 그걸 확정 저장한다.
                if is_prev_entity:
                    word_list.append({"word": _word, "pipeline_label": kpf_to_pipeline(prev_entity_tag)})
                    _word = ""
                _word += token
                is_prev_entity = True
                prev_entity_tag = pred[2:]  # "B-OGG_ECONOMY" → "OGG_ECONOMY"
                is_there_B_before_I = True

            elif 'I-' in pred:
                # B-로 시작된 엔티티의 연속 서브워드. B- 없이 I-만 나온 경우엔
                # is_there_B_before_I가 False라 is_prev_entity가 True로 안 바뀐다.
                _word += token
                if is_there_B_before_I:
                    is_prev_entity = True

            else:  # O: 엔티티가 아닌 토큰 → 누적 중이던 엔티티를 확정 저장하고 리셋
                if is_prev_entity:
                    word_list.append({"word": _word, "pipeline_label": kpf_to_pipeline(prev_entity_tag)})
                    _word = ""
                    is_prev_entity = False
                    is_there_B_before_I = False

        # 청크가 O 없이 엔티티 도중(B-/I-)에 끝나는 경우, 위 루프에선 저장되지
        # 않으므로 [SEP] 직전까지 이어진 마지막 엔티티를 여기서 한 번 더 저장한다.
        if is_prev_entity and _word:
            word_list.append({"word": _word, "pipeline_label": kpf_to_pipeline(prev_entity_tag)})

        return word_list
