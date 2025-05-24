import uuid
from typing import Self

from aikernel import LLMAssistantMessage, LLMUserMessage
from pydantic import BaseModel


class GuideConcept(BaseModel):
    term: str
    definition: str


class Guide(BaseModel):
    id: str
    name: str
    concepts: list[GuideConcept]
    student_learning_style: str
    user_messages: list[LLMUserMessage]
    assistant_messages: list[LLMAssistantMessage]

    @classmethod
    def create(cls, *, name: str, concepts: list[GuideConcept], student_learning_style: str) -> Self:
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            concepts=concepts,
            student_learning_style=student_learning_style,
            user_messages=[],
            assistant_messages=[],
        )

    @property
    def conversation(self) -> list[LLMUserMessage | LLMAssistantMessage]:
        return sorted(self.user_messages + self.assistant_messages, key=lambda message: message.created_at)

    def add_interaction(self, *, user_message: LLMUserMessage, assistant_message: LLMAssistantMessage) -> None:
        self.user_messages.append(user_message)
        self.assistant_messages.append(assistant_message)
