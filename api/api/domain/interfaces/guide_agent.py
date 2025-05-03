from typing import Protocol

from aikernel import LLMAssistantMessage, LLMUserMessage

from api.domain.aggregates.guide import GuideConcept


class IGuideAgent(Protocol):
    async def respond(
        self,
        *,
        concepts: list[GuideConcept],
        student_learning_style: str,
        conversation: list[LLMUserMessage | LLMAssistantMessage],
    ) -> LLMAssistantMessage: ...
