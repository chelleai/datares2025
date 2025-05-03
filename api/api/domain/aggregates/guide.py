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

    @property
    def conversation(self) -> list[LLMUserMessage | LLMAssistantMessage]:
        return sorted(self.user_messages + self.assistant_messages, key=lambda message: message.created_at)

    def add_interaction(self, *, user_message: LLMUserMessage, assistant_message: LLMAssistantMessage) -> None:
        self.user_messages.append(user_message)
        self.assistant_messages.append(assistant_message)
