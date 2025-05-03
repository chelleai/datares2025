from pydantic import BaseModel


class Concept(BaseModel):
    id: str
    term: str
    citations: list[str]
    definition: str
