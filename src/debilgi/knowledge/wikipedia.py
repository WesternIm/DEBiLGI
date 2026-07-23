"""Small read-only client for the MediaWiki Action API."""

from __future__ import annotations

import socket
from typing import Any

import aiohttp

from debilgi.knowledge.models import WikipediaResult


class WikipediaAPIError(RuntimeError):
    """The MediaWiki API returned a structured error instead of search results."""


class WikipediaClient:
    def __init__(
        self,
        *,
        language: str = "ru",
        user_agent: str = "DEBiLGI/0.4 (mailto:local-operator@example.invalid)",
        timeout_seconds: int = 5,
    ) -> None:
        self.language = language
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds
        self._session: aiohttp.ClientSession | None = None

    @property
    def endpoint(self) -> str:
        return f"https://{self.language}.wikipedia.org/w/api.php"

    async def start(self) -> None:
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            connector = aiohttp.TCPConnector(family=socket.AF_INET)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"},
            )

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def search(self, query: str) -> WikipediaResult | None:
        candidates = await self.search_candidates(query)
        return candidates[0] if candidates else None

    async def search_candidates(
        self, query: str, *, limit: int = 5
    ) -> tuple[WikipediaResult, ...]:
        await self.start()
        assert self._session is not None
        parameters = {
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": "0",
            "gsrlimit": str(max(1, min(limit, 10))),
            "prop": "extracts|info|pageprops|links|revisions|categories",
            "exintro": "1",
            "explaintext": "1",
            "inprop": "url",
            "ppprop": "wikibase_item",
            "plnamespace": "0",
            "pllimit": "20",
            "rvprop": "ids",
            "cllimit": "50",
            "redirects": "1",
            "format": "json",
            "formatversion": "2",
            "utf8": "1",
            "maxlag": "5",
        }
        async with self._session.get(self.endpoint, params=parameters) as response:
            response.raise_for_status()
            payload = await response.json(content_type=None)
        return self._parse_candidates(payload)

    @staticmethod
    def _parse(payload: Any) -> WikipediaResult | None:
        candidates = WikipediaClient._parse_candidates(payload)
        return candidates[0] if candidates else None

    @staticmethod
    def _parse_candidates(payload: Any) -> tuple[WikipediaResult, ...]:
        if not isinstance(payload, dict):
            return ()
        error = payload.get("error")
        if isinstance(error, dict):
            code = str(error.get("code") or "unknown")
            information = str(error.get("info") or "MediaWiki request failed")
            raise WikipediaAPIError(f"{code}: {information}")
        query = payload.get("query")
        pages = query.get("pages") if isinstance(query, dict) else None
        if not isinstance(pages, list) or not pages:
            return ()
        parsed = tuple(
            result
            for page in pages
            if isinstance(page, dict)
            if (result := WikipediaClient._parse_page(page)) is not None
        )
        return parsed

    @staticmethod
    def _parse_page(page: dict[str, Any]) -> WikipediaResult | None:
        if not isinstance(page, dict) or page.get("missing") is True:
            return None
        page_id = page.get("pageid")
        title = page.get("title")
        extract = page.get("extract")
        if (
            not isinstance(page_id, int)
            or not isinstance(title, str)
            or not isinstance(extract, str)
        ):
            return None
        revisions = page.get("revisions")
        revision_id = None
        if isinstance(revisions, list) and revisions and isinstance(revisions[0], dict):
            value = revisions[0].get("revid")
            revision_id = value if isinstance(value, int) else None
        pageprops = page.get("pageprops")
        wikidata_id = pageprops.get("wikibase_item") if isinstance(pageprops, dict) else None
        raw_links = page.get("links")
        links = raw_links if isinstance(raw_links, list) else []
        related_titles = tuple(
            linked_title
            for item in links
            if isinstance(item, dict)
            if isinstance((linked_title := item.get("title")), str) and ":" not in linked_title
        )[:20]
        canonical_url = page.get("canonicalurl")
        if not isinstance(canonical_url, str):
            canonical_url = ""
        raw_categories = page.get("categories")
        categories = tuple(
            title
            for item in raw_categories
            if isinstance(item, dict)
            if isinstance((title := item.get("title")), str)
        ) if isinstance(raw_categories, list) else ()
        return WikipediaResult(
            page_id=page_id,
            title=title,
            extract=extract,
            canonical_url=canonical_url,
            wikidata_id=wikidata_id if isinstance(wikidata_id, str) else None,
            revision_id=revision_id,
            related_titles=related_titles,
            categories=categories,
        )
