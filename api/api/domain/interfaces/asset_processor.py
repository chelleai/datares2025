from typing import Protocol

from pydantic import BaseModel


class IdentifiedConcept(BaseModel):
    term: str
    citations: list[str]
    definition: str


class IAssetProcessor(Protocol):
    async def process(self, *, content: str) -> list[IdentifiedConcept]: ...
