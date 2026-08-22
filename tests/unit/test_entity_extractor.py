from app.graph.nodes.entity_extractor import GAZETTEERS, EntityExtractor


def test_only_the_company_gazetteer_is_loaded():
    assert list(GAZETTEERS.keys()) == ["COMPANY"]


def test_extracts_companies_from_text():
    extractor = EntityExtractor()

    entities = extractor.extract("삼성전자가 SK하이닉스와 경쟁한다.")

    texts = {entity.text for entity in entities}
    assert "삼성전자" in texts
    assert "SK하이닉스" in texts


def test_products_are_no_longer_extracted():
    extractor = EntityExtractor()

    entities = extractor.extract("삼성전자가 HBM과 니켈을 다룬다.")

    texts = {entity.text for entity in entities}
    assert "HBM" not in texts
    assert "니켈" not in texts


def test_canonicalize_leaves_product_nouns_untouched():
    extractor = EntityExtractor()

    # "조립" used to be rewritten by the product gazetteer, corrupting the sentence.
    assert extractor.canonicalize("배터리 조립라인") == "배터리 조립라인"
