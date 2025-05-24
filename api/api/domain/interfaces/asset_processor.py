from typing import Protocol

from pydantic import BaseModel

from api.domain.aggregates.asset import Concept


class CitedTerm(BaseModel):
    term: str
    citations: list[str]


class IAssetProcessor(Protocol):
    async def identify_terms(self, *, content: str) -> list[CitedTerm]: ...
    async def synthesize_definition(self, *, cited_term: CitedTerm) -> str: ...
    async def process(self, *, content: str) -> list[Concept]: ...
