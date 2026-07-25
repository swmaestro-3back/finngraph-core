"""
현재 KPF-bert-ner 모델은 내부적으로 한 문장씩 청킹하여 처리한다.
따라서 문단을 넣어도 내부적으로 split sentences를 통해
문장 분할을 하고 for sent in sentences로 NER 작업을 진행한다.
따라서 현 프로젝트에서는 형태소 분할을 하는 kiwi도 한 문장 단위로 진행되어 split sentences를 하는 점을 파악하여
형태소분할할때 진행된 분할된 문장들을 그대로 사용하여 현재 NER 함수에는 하나의 문장이 입력으로 들어온다.
"""
import torch
from transformers import AutoTokenizer, BertForTokenClassification, logging as hf_logging

from app.graph.utils.kpf import ID2KPF, kpf_to_finngraph_label
from app.graph.models import Entity

# transformers 라이브러리 자체의 로그 레벨을 ERROR로 올려서,
# 평소 INFO/WARNING 레벨로 출력되는 로딩 관련 메시지들을 조용히 시키는 용도
hf_logging.set_verbosity_error()

# ner 호출이 배치 처리로 그래프 실행당 1번뿐이라 예전처럼 fan-out과 intra-op 병렬화가
# 곱해져 오버서브스크립션이 생기는 경우는 없다. 이 머신(8 성능 + 2 효율 코어) 기준
# 실측상 8이 가장 빨라 intra-op 스레드 수를 8로 고정한다.
# (docs/ner_batch_migration.md 참고)
torch.set_num_threads(8)

MODEL_PATH = "./KPF-bert-ner"
# chunking 노드가 이미 512자 이내로 잘라서 넘겨주므로 문장 단위 재분할 없이 청크를
# 통째로 추론한다. 모델의 max_position_embeddings가 512라 truncation으로 안전장치를 둔다.
MAX_LENGTH = 512

class NER:
    def __init__(self):
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self._model = BertForTokenClassification.from_pretrained(MODEL_PATH)
        self._model.eval()

    # texts: chunking 노드가 만든 문장(청크)들의 리스트. 문장 개수만큼 fan-out 하는 대신
    # 하나의 배치로 묶어 forward pass를 한 번만 호출한다 (docs/ner_fanout_thread_contention.md).
    def extract_entities_batch(self, texts: list[str]) -> list[list[Entity]]:
        batch_kpf_entries = self._ner_predict_batch(texts)
        return [self._to_entities(kpf_entries) for kpf_entries in batch_kpf_entries]

    def _to_entities(self, kpf_entries: list[dict]) -> list[Entity]:
        entities: list[Entity] = []

        for entry in kpf_entries:
            pipeline_label = entry["pipeline_label"]
            # pipeline_label=None: 파이프라인 태그셋(기업 간 관계)에 대응 없는 KPF 카테고리 → 제외
            if pipeline_label is None:
                continue
            entities.append(Entity(text=entry["word"], label=pipeline_label))

        return entities

    # 배치처리용이므로 list[str]으로 문장 리스트를 인자로 받는다.
    def _ner_predict_batch(self, texts: list[str]) -> list[list[dict]]:
        """KPF 공식 ner_module.py 로직 기반 (CPU 버전). 문장 여러 개를 하나의 배치로 추론한다."""
        # KPF 모델 학습 시 공백을 '-'로 치환했으므로 추론 시에도 동일하게 적용
        texts_dashed = [text.replace(" ", "-") for text in texts]

        inputs = self._tokenizer(
            texts_dashed,           # 이 텍스트를 토큰화하고
            return_tensors="pt",    # 모델이 이해할 수 있는 pytorch tensor(pt) 형태로 변환하고
            padding=True,           #
            truncation=True,        # 최대길이를 넘어가면 잘라버려라
            max_length=MAX_LENGTH,
        )

        # 기울기 계산 끄기 - 학습 단계가 아닌 추론/예측 단계이므로
        with torch.no_grad():
            outputs = self._model(**inputs) # 토큰화된 데이터들을 모델에 넣어 예측 결과 받기

        # [batch_size, seq_len] — 문장마다 독립적으로 계산된 예측. 배치 차원은 attention이
        # 넘나들지 않으므로 문장끼리 서로 영향을 주지 않는다.
        token_predictions = outputs.logits.argmax(dim=2).tolist()
        attention_mask = inputs["attention_mask"].tolist()
        batch_tokens = [encoding.tokens for encoding in inputs.encodings]

        return [
            self._merge_bio(tokens, [ID2KPF[label_id] for label_id in preds], mask)
            for tokens, preds, mask in zip(batch_tokens, token_predictions, attention_mask)
        ]

    # 아래 로직은 토큰(서브워드) 단위 BIO 예측을 엔티티 단위 텍스트로 모으는
    # 상태 머신이다. 모델은 "삼성전자" 같은 한 단어도 서브워드로 쪼개어
    # B-OGG_ECONOMY / I-OGG_ECONOMY / I-OGG_ECONOMY 식으로 토큰마다 예측하므로,
    # 이어붙여서 하나의 엔티티로 복원하는 과정이 반드시 필요하다.
    def _merge_bio(self, tokens: list[str], pred_str: list[str], attention_mask: list[int]) -> list[dict]:
        # 배치 추론 시 짧은 문장은 뒤(혹은 앞)에 [PAD]가 채워지므로, attention_mask로
        # 실제 토큰만 걸러내 이 문장 몫이 아닌 패딩 예측이 섞이지 않게 한다.
        real_tokens = [t for t, m in zip(tokens, attention_mask) if m == 1]
        real_preds = [p for p, m in zip(pred_str, attention_mask) if m == 1]

        word_list: list[dict] = []

        is_prev_entity = False
        prev_entity_tag = ""
        # 모델이 B- 없이 I-로 엔티티를 시작하는(BIO 규칙 위반) 경우를 방어한다.
        # 이런 경우 _word는 계속 누적되지만 is_prev_entity=False로 유지되어,
        # 다음 O/B- 시점에 엔티티로 저장되지 않고 조용히 버려진다.
        is_there_B_before_I = False
        _word = ""

        # real_tokens[0]/[-1]: [CLS]/[SEP] — 특수 토큰이므로 건너뜀
        for token, pred in zip(real_tokens[1:-1], real_preds[1:-1]):
            # '##' 제거로 서브워드 복원, '-'는 원래 공백이었으므로 되돌림
            token = token.replace('#', '').replace("-", " ")
            if token == "":
                continue

            if 'B-' in pred:
                # 새 엔티티의 시작. 직전에 다른 엔티티를 누적 중이었다면
                # (연속된 두 엔티티 사이에 O가 안 끼는 경우) 먼저 그걸 확정 저장한다.
                if is_prev_entity:
                    word_list.append({"word": _word, "pipeline_label": kpf_to_finngraph_label(prev_entity_tag)})
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
                    word_list.append({"word": _word, "pipeline_label": kpf_to_finngraph_label(prev_entity_tag)})
                    _word = ""
                    is_prev_entity = False
                    is_there_B_before_I = False

        # 청크가 O 없이 엔티티 도중(B-/I-)에 끝나는 경우, 위 루프에선 저장되지
        # 않으므로 [SEP] 직전까지 이어진 마지막 엔티티를 여기서 한 번 더 저장한다.
        if is_prev_entity and _word:
            word_list.append({"word": _word, "pipeline_label": kpf_to_finngraph_label(prev_entity_tag)})

        return word_list
