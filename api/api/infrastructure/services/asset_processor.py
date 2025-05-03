from typing import Self

from aikernel import LLMMessagePart, LLMModelName, LLMSystemMessage, LLMUserMessage, get_router, llm_structured
from pydantic import BaseModel

from api.domain.aggregates.asset import Concept
from api.domain.interfaces.asset_processor import CitedTerm, IAssetProcessor


class CitedTermExtraction(BaseModel):
    cited_terms: list[CitedTerm]


class DefinitionSynthesis(BaseModel):
    definition: str


class AssetProcessor(IAssetProcessor):
    def __call__(self) -> Self:
        return self  # trick to help with validation against the interface

    async def identify_terms(self, *, content: str) -> list[CitedTerm]:
        """YOUR IMPLEMENTATION GOES IN THIS FUNCTION.

        This is step 1 of the asset processing pipeline.
        The input `content` is the Markdown content of an uploaded file.
        The output is a list of `CitedTerm` objects, which contain a term and a list of citations.

        The goal is for term extraction to extract 100% of the relevant terms in the content, with
        minimal false positives or false negatives, and for the citations to be meaningful and accurate.

        Instructions for prompting:
        - Prefer the Gemini series of models if possible.
        - Make use of the `content` in the prompt.
        - You can add as many messages to the array as you want; this may be helpful for techniques such as few-shot prompting.
        """
        ai_response = await llm_structured(
            router=get_router(models=(LLMModelName.GEMINI_20_FLASH,)),
            messages=[
                LLMSystemMessage(parts=[LLMMessagePart(content="This is the system prompt...")]),
                LLMUserMessage(parts=[LLMMessagePart(content="This is the user prompt...")]),
            ],
            response_model=CitedTermExtraction,
        )

        return [
            CitedTerm(
                term=cited_term.term,
                citations=cited_term.citations,
            )
            for cited_term in ai_response.structured_response.cited_terms
        ]

    async def synthesize_definition(self, *, cited_term: CitedTerm) -> str:
        """YOUR IMPLEMENTATION GOES IN THIS FUNCTION.

        This is step 2 of the asset processing pipeline.
        The input `cited_term` is a `CitedTerm` object, which contains a term and a list of citations.
        The output is a definition of the term, synthesized from the citations.

        Instructions for prompting:
        - Prefer the Gemini series of models if possible.
        - Make use of the `cited_term` properties in the prompt.
            - You can find the schema of the CitedTerm object in `domain/interfaces/asset_processor.py`.
        - You can add as many messages to the array as you want; this may be helpful for techniques such as few-shot prompting.
        """
        ai_response = await llm_structured(
            router=get_router(models=(LLMModelName.GEMINI_20_FLASH,)),
            messages=[
                LLMSystemMessage(parts=[LLMMessagePart(content="This is the system prompt...")]),
                LLMUserMessage(parts=[LLMMessagePart(content="This is the user prompt...")]),
            ],
            response_model=DefinitionSynthesis,
        )

        return ai_response.structured_response.definition

    async def process(self, *, content: str) -> list[Concept]:
        """This function should not change. This is not a prompting task."""
        cited_terms = await self.identify_terms(content=content)
        definitions = [await self.synthesize_definition(cited_term=cited_term) for cited_term in cited_terms]
        return [
            Concept(
                term=cited_term.term,
                definition=definition,
                citations=cited_term.citations,
            )
            for cited_term, definition in zip(cited_terms, definitions)
        ]
