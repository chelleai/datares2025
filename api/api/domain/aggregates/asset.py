from pydantic import BaseModel


class Concept(BaseModel):
    term: str
    citations: list[str]
    definition: str


class Asset(BaseModel):
    id: str
    name: str
    content: str
    concepts: list[Concept]
