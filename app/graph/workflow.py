from kiwipiepy import Sentence
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from app.graph.models import Entity
from app.graph.state import GraphState
from app.graph.nodes.cleaning import Cleaner
from app.graph.nodes.morphology import MorphAnalyzer
from app.graph.nodes.ner import NER
from app.graph.nodes.srl import SRL
from app.graph.nodes.fpdf import FPDF

class GraphRunner:
    def __init__(self):
        self._cleaner = Cleaner()
        self._morph_analyzer = MorphAnalyzer()
        self._ner = NER()
        self._srl = SRL()
        self._fpdf = FPDF()
        self._graph = self._compile_graph(
            self._cleaner,
            self._morph_analyzer,
            self._ner,
            self._srl,
            self._fpdf,
        )

    def _compile_graph(
        self,
        cleaner: Cleaner,
        morph_analyzer: MorphAnalyzer,
        ner: NER,
        srl: SRL,
        fpdf: FPDF,
    ):
        """cleaning ~ fpdf 노드를 잇는 LangGraph 워크플로우를 빌드하고 컴파일한다."""

        def cleaning_node(state: GraphState) -> dict:
            cleaned = cleaner.clean(state["article"])
            return {"article": cleaned}

        def chunking_node(state: GraphState) -> dict:
            """
            [Chunking Node]
            Cleaner에서 NER을 수행하기 전에, kiwi로 문단을 문장 단위로 분할해 청크를 생성한다.
            kiwi의 문장 분리는 내부적으로 형태소 분석을 수반하므로, 이 결과(tokens 포함)를
            morphology 노드가 그대로 재사용해 형태소 분석이 중복 수행되지 않도록 한다.
            """
            sentences : list[Sentence] = morph_analyzer.split_sentences(state["article"])
            return {"sentences": sentences}

        def route_to_ner(state: GraphState):
            # 문장 개수만큼 NER 노드를 동적으로 병렬 실행하도록 Send 객체를 반환한다
            return [Send("ner", {"chunk_text": sent.text}) for sent in state["sentences"]]

        def morphology_node(state: GraphState) -> dict:
            verb_lemmas = morph_analyzer.analyze(state["sentences"])
            return {"verb_lemmas": verb_lemmas}

        def ner_node(state: GraphState) -> dict:
            # Send로 전달받은 청크 하나에 대해서만 NER을 돌린다. 여러 ner 노드 인스턴스가
            # 동시에 실행되므로 결과는 ner_chunks 리듀서 채널에 리스트로 이어붙인다.
            entities = ner.extract_entities(state["chunk_text"])
            return {"ner_chunks": [entities]}

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

        workflow.add_node("cleaning", cleaning_node)
        workflow.add_node("chunking", chunking_node)
        workflow.add_node("morphology", morphology_node)
        workflow.add_node("ner", ner_node)
        workflow.add_node("merge_ner", merge_ner_node)
        workflow.add_node("srl", srl_node)
        workflow.add_node("fpdf", fpdf_node)

        workflow.set_entry_point("cleaning")

        workflow.add_edge("cleaning", "chunking")

        # FAN-OUT: chunking에서 분리된 문장들을 NER(Send 기반 동적 라우팅)과 morphology가 공유해 사용
        workflow.add_conditional_edges("chunking", route_to_ner, ["ner"])
        workflow.add_edge("chunking", "morphology")

        # 병렬 실행된 ner 인스턴스가 모두 끝나야 merge_ner로 진입 (fan-in)
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