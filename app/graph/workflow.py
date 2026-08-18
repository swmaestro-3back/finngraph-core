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

        async def normalize_article(state: GraphState) -> dict:
            """gazetteer를 토대로 등록된 엔티티 명칭 표준화"""
            normalized_article = await asyncio.to_thread(entity_extractor.normalize, state["article"])
            return {"article": normalized_article}

        async def extract_entities(state: GraphState) -> dict:
            """gazetteer를 바탕으로 엔티티 추출"""
            gazetteer_entities = await asyncio.to_thread(
                entity_extractor.extract_entities, state["article"]
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
            """관계 후보 프레임만 추출한다 (의미 라벨은 다음 노드에서 붙인다)."""
            candidates = await relation_extractor.label(state["article"], state["entities"])
            return {"candidates": candidates}

        async def annotate_frames(state: GraphState) -> dict:
            """원문 전체를 다시 보며 evidence/polarity/tense를 붙인다."""
            relations, annotation_stats = await frame_annotator.annotate(
                state["article"], state["candidates"]
            )
            return {"relations": relations, "annotation_stats": annotation_stats}

        async def build_triplets(state: GraphState) -> dict:
            relations = state["relations"]
            return {
                "triplets": triplet_builder.filter(relations),
                "triplet_stats": merge_stats(
                    triplet_builder.stats(relations),
                    state.get("annotation_stats", {}),
                ),
            }

        workflow = StateGraph(GraphState)

        workflow.add_node("normalizer", normalize_article)
        workflow.add_node("entity_extractor", extract_entities)
        workflow.add_node("relation_extractor", extract_relations)
        workflow.add_node("frame_annotator", annotate_frames)
        workflow.add_node("triplet_builder", build_triplets)

        workflow.set_entry_point("normalizer")

        workflow.add_edge("normalizer", "entity_extractor")
        workflow.add_edge("entity_extractor", "relation_extractor")
        workflow.add_edge("relation_extractor", "frame_annotator")
        workflow.add_edge("frame_annotator", "triplet_builder")
        workflow.add_edge("triplet_builder", END)

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
