import uuid

from pydantic import BaseModel

from api.domain.aggregates.asset.entities import Concept


class Asset(BaseModel):
    id: str
    name: str
    content: str
    concepts: list[Concept]

    @classmethod
    def create(cls, *, name: str, content: str) -> "Asset":
        return cls(id=str(uuid.uuid4()), name=name, content=content, concepts=[])

    def add_concept(self, *, term: str, citations: list[str], definition: str) -> None:
        self.concepts.append(Concept(id=str(uuid.uuid4()), term=term, citations=citations, definition=definition))
