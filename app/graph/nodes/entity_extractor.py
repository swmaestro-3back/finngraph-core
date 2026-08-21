from flashtext import KeywordProcessor

from app.graph.models import Entity, EntityLabel
from app.graph.ontology.gazetteers import COUNTRY_DICT, COMMODITY_DICT, PRODUCT_DICT, COMPANY_DICT

# Pre-built knowledge base dict
GAZETTEERS: dict[EntityLabel, dict[str, list[str]]] = {
    "COMPANY": COMPANY_DICT,
    "COUNTRY": COUNTRY_DICT,
    "PRODUCT": PRODUCT_DICT,
    "COMMODITY": COMMODITY_DICT
}

class EntityExtractor:
    def __init__(self):
    
        self._canonicalizer = KeywordProcessor(case_sensitive=True)
        self._processors: dict[EntityLabel, KeywordProcessor] = {}

        for label, gazetteer in GAZETTEERS.items():
            processor = KeywordProcessor(case_sensitive=True)
            processor.add_keywords_from_dict(gazetteer)
            self._canonicalizer.add_keywords_from_dict(gazetteer)
            self._processors[label] = processor

    def canonicalize(self, text: str) -> str:
        """
        Replace gazetteer surface forms with their canonical names
        """
        return self._canonicalizer.replace_keywords(text)

    def extract(self, text: str) -> list[Entity]:
        """
        Extract entities using gazetteer
        """
        entities: list[Entity] = []
        for label, processor in self._processors.items():
            for canonical in processor.extract_keywords(text):
                entities.append(Entity(text=canonical, label=label))
        return entities
