import asyncio

from langgraph.graph import END, StateGraph

from app.graph.state import GraphState
from app.graph.models import Entity
from app.graph.nodes.relation_extractor import RelationExtractor
from app.graph.nodes.frame_annotator import FrameAnnotator
from app.graph.nodes.triplet_builder import TripletBuilder
from app.graph.nodes.entity_extractor import EntityExtractor


def merge_stats(triplet_stats: dict, annotation_stats: dict) -> dict:
    """TripletBuilder 통계와 FrameAnnotator 드롭 카운트를 하나의 dict로 합친다.

    드롭은 FrameAnnotator에서 발생하지만 stats()는 살아남은 프레임만 보기 때문에,
    두 출처를 여기서 합쳐야 손실률이 보인다.
    """

    merged = dict(triplet_stats)
    merged["dropped_annotation_mismatch"] = annotation_stats.get("dropped_annotation_mismatch", 0)
    merged["dropped_evidence_grounding"] = annotation_stats.get("dropped_evidence_grounding", 0)
    return merged


class GraphRunner:
    def __init__(self):
        self._entity_extractor = EntityExtractor()
        self._relation_extractor = RelationExtractor()
        self._frame_annotator = FrameAnnotator()
        self._triplet_builder = TripletBuilder()
        self._graph = self._compile_graph(
            self._entity_extractor,
            self._relation_extractor,
            self._frame_annotator,
            self._triplet_builder,
        )

    def _compile_graph(
        self,
        entity_extractor: EntityExtractor,
        relation_extractor: RelationExtractor,
        frame_annotator: FrameAnnotator,
        triplet_builder: TripletBuilder,
    ):

        async def canonicalize_article(state: GraphState) -> dict:
            """Replace gazetteer surface forms in the article with canonical names."""
            canonicalized_article = await asyncio.to_thread(
                entity_extractor.canonicalize, state["article"]
            )
            return {"article": canonicalized_article}

        async def extract_entities(state: GraphState) -> dict:
            """Extract entities based on pre-built knowledge base"""
            gazetteer_entities = await asyncio.to_thread(
                entity_extractor.extract, state["article"]
            )

            seen: set[tuple[str, str]] = set()
            deduped: list[Entity] = []
            for entity in gazetteer_entities:
                key = (entity.text, entity.label)
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(entity)
            return {"entities": deduped}

        async def extract_relations(state: GraphState) -> dict:
            """Extract relation frame candidates by refering to base ontology"""
            candidate_frames = await relation_extractor.extract(
                state["article"], state["entities"]
            )
            return {"candidate_frames": candidate_frames}

        async def annotate_frames(state: GraphState) -> dict:
            """Define evidence/polarity/tense refering to the article"""
            annotated_frames, annotation_stats = await frame_annotator.annotate(
                state["article"], state["candidate_frames"]
            )
            return {
                "annotated_frames": annotated_frames,
                "annotation_stats": annotation_stats,
            }

        async def build_triplets(state: GraphState) -> dict:
            annotated_frames = state["annotated_frames"]
            return {
                "triplets": triplet_builder.build(annotated_frames),
                "triplet_stats": merge_stats(
                    triplet_builder.stats(annotated_frames),
                    state.get("annotation_stats", {}),
                ),
            }

        workflow = StateGraph(GraphState)

        # 노드 ID를 래퍼 함수명과 동일하게 둔다. 이 ID가 LangSmith 트레이스에 스텝 이름으로
        # 그대로 노출되므로, 한 스텝에 이름이 두 개 생기지 않도록 맞춘다.
        workflow.add_node("canonicalize_article", canonicalize_article)
        workflow.add_node("extract_entities", extract_entities)
        workflow.add_node("extract_relations", extract_relations)
        workflow.add_node("annotate_frames", annotate_frames)
        workflow.add_node("build_triplets", build_triplets)

        workflow.set_entry_point("canonicalize_article")

        workflow.add_edge("canonicalize_article", "extract_entities")
        workflow.add_edge("extract_entities", "extract_relations")
        workflow.add_edge("extract_relations", "annotate_frames")
        workflow.add_edge("annotate_frames", "build_triplets")
        workflow.add_edge("build_triplets", END)

        return workflow.compile()

    async def ainvoke(self, news_id: str, article: str) -> GraphState:
        """LangGraph 워크플로우를 비동기로 실행한다."""
        final_state = await self._graph.ainvoke(
            GraphState(
                news_id=news_id,
                article=article,
            )
        )
        return final_state
