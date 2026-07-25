from kiwipiepy import Kiwi
from langgraph.graph import END, StateGraph

from app.graph.state import GraphState
from app.graph.models import Entity
from app.graph.nodes.ner import NER
from app.graph.nodes.srl import SRL
from app.graph.nodes.fpdf import FPDF

class GraphRunner:
    def __init__(self):
        self._kiwi = Kiwi()
        self._ner = NER()
        self._srl = SRL()
        self._fpdf = FPDF()
        self._graph = self._compile_graph(
            self._kiwi,
            self._ner,
            self._srl,
            self._fpdf,
        )

    def _compile_graph(
        self,
        kiwi: Kiwi,
        ner: NER,
        srl: SRL,
        fpdf: FPDF,
    ):
        """chunking ~ fpdf 노드를 잇는 LangGraph 워크플로우를 빌드하고 컴파일한다."""

        def chunking_node(state: GraphState) -> dict:
            """
            [Chunking Node]
            NER을 수행하기 전에, kiwi로 문단을 문장 단위로 분할해 청크를 생성한다.
            """
            sentences = kiwi.split_into_sents(state["article"], return_tokens=False)
            return {"sentences": [sent.text for sent in sentences]}

        def ner_node(state: GraphState) -> dict:
            # 문장별로 fan-out 하는 대신, 문장 리스트 전체를 배치로 묶어 한 번에 추론한다.
            entities_per_sentence = ner.extract_entities_batch(state["sentences"])
            # 청크 경계에 걸쳐 중복 추출된 (text, label)을 앞에서 나온 순서대로 하나만 남긴다.
            seen: set[tuple[str, str]] = set()
            deduped: list[Entity] = []
            for chunk_entities in entities_per_sentence:
                for entity in chunk_entities:
                    key = (entity.text, entity.label)
                    if key in seen:
                        continue
                    seen.add(key)
                    deduped.append(entity)
            return {"ner": deduped}

        def srl_node(state: GraphState) -> dict:
            srl_output = srl.label(state["article"], state["ner"])
            return {"srl": srl_output}

        def fpdf_node(state: GraphState) -> dict:
            srl_output = state["srl"]
            return {"triples": fpdf.filter(srl_output), "fpdf_stats": fpdf.stats(srl_output)}

        workflow = StateGraph(GraphState)

        workflow.add_node("chunking", chunking_node)
        workflow.add_node("ner", ner_node)
        workflow.add_node("srl", srl_node)
        workflow.add_node("fpdf", fpdf_node)

        workflow.set_entry_point("chunking")

        workflow.add_edge("chunking", "ner")
        workflow.add_edge("ner", "srl")
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