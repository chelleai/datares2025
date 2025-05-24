from typing import Self

from aikernel import (
    LLMAssistantMessage,
    LLMMessagePart,
    LLMModelName,
    LLMSystemMessage,
    LLMUserMessage,
    get_router,
    llm_unstructured,
)

from api.domain.aggregates.guide import GuideConcept
from api.domain.interfaces.guide_agent import IGuideAgent


class GuideAgent(IGuideAgent):
    def __call__(self) -> Self:
        return self  # trick to help with validation against the interface

    async def respond(
        self,
        *,
        concepts: list[GuideConcept],
        student_learning_style: str,
        conversation: list[LLMUserMessage | LLMAssistantMessage],
    ) -> LLMAssistantMessage:
        """YOUR IMPLEMENTATION GOES IN THIS FUNCTION.

        This is the method that Guides will use to respond to each student message.
        The available context is:
        - `concepts`: A list of `GuideConcept` objects, which contain a term and a definition.
            - These are the concepts that the Guide is trying to teach the student about.
        - `student_learning_style`: Text that the student has written that describes how they prefer to learn.
        - `conversation`: An ordered list of messages between the student and the Guide so far.
            - Likely you will want to use these messages by prepending them to the conversation when invoking the LLM.

        Instructions for prompting:
        - Prefer the Gemini series of models if possible.
        - Make use of the `concepts`, `student_learning_style`, and `conversation` in the prompt.
        """
        ai_response = await llm_unstructured(
            router=get_router(models=(LLMModelName.GEMINI_20_FLASH,)),
            messages=[
                LLMSystemMessage(parts=[LLMMessagePart(content="This is the system prompt...")]),
                LLMUserMessage(parts=[LLMMessagePart(content="This is the user prompt...")]),
            ],
        )

        return LLMAssistantMessage(parts=[LLMMessagePart(content=ai_response.text)])
