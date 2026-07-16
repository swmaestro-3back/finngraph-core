from app.graph.models import Entity

# (text, label)이 같은 엔티티를 앞에서 나온 순서대로 하나만 남긴다.
def dedupe_entities(entities: list[Entity]) -> list[Entity]:
    seen: set[tuple[str, str]] = set()
    deduped: list[Entity] = []
    for entity in entities:
        key = (entity.text, entity.label)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entity)
    return deduped
