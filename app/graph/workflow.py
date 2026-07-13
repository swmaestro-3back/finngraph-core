from kiwipiepy import Sentence
from langgraph.graph import END, StateGraph

from app.graph.models import Entity
from app.graph.state import GraphState
from app.graph.nodes.morphology import MorphAnalyzer
from app.graph.nodes.ner import NER
from app.graph.nodes.srl import SRL
from app.graph.nodes.fpdf import FPDF

class GraphRunner:
    def __init__(self):
        self._morph_analyzer = MorphAnalyzer()
        self._ner = NER()
        self._srl = SRL()
        self._fpdf = FPDF()
        self._graph = self._compile_graph(
            self._morph_analyzer,
            self._ner,
            self._srl,
            self._fpdf,
        )

    def _compile_graph(
        self,
        morph_analyzer: MorphAnalyzer,
        ner: NER,
        srl: SRL,
        fpdf: FPDF,
    ):
        """chunking ~ fpdf 노드를 잇는 LangGraph 워크플로우를 빌드하고 컴파일한다."""

        def chunking_node(state: GraphState) -> dict:
            """
            [Chunking Node]
            NER을 수행하기 전에, kiwi로 문단을 문장 단위로 분할해 청크를 생성한다.
            kiwi의 문장 분리는 내부적으로 형태소 분석을 수반하므로, 이 결과(tokens 포함)를
            morphology 노드가 그대로 재사용해 형태소 분석이 중복 수행되지 않도록 한다.
            """
            sentences : list[Sentence] = morph_analyzer.split_sentences(state["article"])
            return {"sentences": sentences}

        def morphology_node(state: GraphState) -> dict:
            verb_lemmas = morph_analyzer.analyze(state["sentences"])
            return {"verb_lemmas": verb_lemmas}

        def ner_node(state: GraphState) -> dict:
            # 문장별로 fan-out 하는 대신, 문장 리스트 전체를 배치로 묶어 한 번에 추론한다.
            entities_per_sentence = ner.extract_entities_batch(
                [sent.text for sent in state["sentences"]]
            )
            return {"ner_chunks": entities_per_sentence}

        def merge_ner_node(state: GraphState) -> dict:
            # 청크 경계에 걸쳐 중복 추출된 (text, label)을 제거하며 하나의 리스트로 합친다
            seen: set[tuple[str, str]] = set()
            merged: list[Entity] = []
            for chunk_entities in state["ner_chunks"]:
                for entity in chunk_entities:
                    key = (entity.text, entity.label)
                    if key in seen:
                        continue
                    seen.add(key)
                    merged.append(entity)
            return {"ner": merged}

        def srl_node(state: GraphState) -> dict:
            srl_output = srl.label(state["article"], state["ner"], state["verb_lemmas"])
            return {"srl": srl_output}

        def fpdf_node(state: GraphState) -> dict:
            srl_output = state["srl"]
            return {"triples": fpdf.filter(srl_output), "fpdf_stats": fpdf.stats(srl_output)}

        workflow = StateGraph(GraphState)

        workflow.add_node("chunking", chunking_node)
        workflow.add_node("morphology", morphology_node)
        workflow.add_node("ner", ner_node)
        workflow.add_node("merge_ner", merge_ner_node)
        workflow.add_node("srl", srl_node)
        workflow.add_node("fpdf", fpdf_node)

        workflow.set_entry_point("chunking")

        # FAN-OUT: chunking에서 분리된 문장들을 ner(배치 추론)과 morphology가 공유해 사용
        workflow.add_edge("chunking", "ner")
        workflow.add_edge("chunking", "morphology")

        workflow.add_edge("ner", "merge_ner")

        # FAN-IN: morphology와 merge_ner 노드가 모두 완료되어야 SRL 노드로 진입 가능
        workflow.add_edge(["morphology", "merge_ner"], "srl")
        workflow.add_edge("srl", "fpdf")
        workflow.add_edge("fpdf", END)

        return workflow.compile()

    def invoke(self, article_id: str, article: str) -> GraphState:
        """LangGraph 워크플로우를 실행한다."""
        final_state = self._graph.invoke(
            GraphState(
                article_id=article_id,
                article=article,
            )
        )
        return final_state