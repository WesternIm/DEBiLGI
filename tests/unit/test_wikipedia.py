from debilgi.knowledge.wikipedia import WikipediaClient


def test_wikipedia_payload_is_parsed_into_a_concept_result() -> None:
    result = WikipediaClient._parse(
        {
            "query": {
                "pages": [
                    {
                        "pageid": 123,
                        "title": "Дракон",
                        "extract": "Дракон — мифическое существо.",
                        "canonicalurl": "https://ru.wikipedia.org/wiki/Дракон",
                        "pageprops": {"wikibase_item": "Q7559"},
                        "revisions": [{"revid": 777}],
                        "links": [{"title": "Мифология"}, {"title": "Категория:Драконы"}],
                    }
                ]
            }
        }
    )

    assert result is not None
    assert result.page_id == 123
    assert result.wikidata_id == "Q7559"
    assert result.revision_id == 777
    assert result.related_titles == ("Мифология",)


def test_wikipedia_payload_without_pages_has_no_result() -> None:
    assert WikipediaClient._parse({"query": {}}) is None
