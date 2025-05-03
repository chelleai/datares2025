from aikernel import LLMMessagePart, LLMUserMessage
from fastapi import APIRouter, Depends

from api.application.guides.models import (
    CreateGuideRequest,
    CreateGuideResponse,
    CreateMessageRequest,
    CreateMessageResponse,
    GuideOverview,
    ListGuidesResponse,
)
from api.domain.aggregates.guide import Guide
from api.domain.interfaces.guide_agent import IGuideAgent
from api.domain.repositories import IGuideRepository
from api.infrastructure.repositories import GuideRepository
from api.infrastructure.services.guide_agent import GuideAgent

router = APIRouter(prefix="/guides", tags=["Guides"])


@router.get("", response_model=ListGuidesResponse)
async def list_guides(
    guide_repository: IGuideRepository = Depends(GuideRepository()),
) -> ListGuidesResponse:
    guides = await guide_repository.list()
    overviews = [GuideOverview(id=guide.id, name=guide.name, concepts=guide.concepts) for guide in guides]
    return ListGuidesResponse(guides=overviews)


@router.get("/{guide_id}", response_model=Guide)
async def get_guide(
    guide_id: str,
    guide_repository: IGuideRepository = Depends(GuideRepository()),
) -> Guide:
    return await guide_repository.get(id=guide_id)


@router.post("", response_model=CreateGuideResponse)
async def create_guide(
    request: CreateGuideRequest,
    guide_repository: IGuideRepository = Depends(GuideRepository()),
) -> CreateGuideResponse:
    guide = Guide.create(
        name=request.name, concepts=request.concepts, student_learning_style=request.student_learning_style
    )
    await guide_repository.save(guide=guide)
    return CreateGuideResponse(id=guide.id)


@router.post("/{guide_id}/messages", response_model=CreateMessageResponse)
async def create_message(
    guide_id: str,
    request: CreateMessageRequest,
    guide_repository: IGuideRepository = Depends(GuideRepository()),
    guide_agent: IGuideAgent = Depends(GuideAgent()),
) -> CreateMessageResponse:
    guide = await guide_repository.get(id=guide_id)

    message = LLMUserMessage(parts=[LLMMessagePart(content=request.message)])
    response = await guide_agent.respond(
        concepts=guide.concepts,
        student_learning_style=guide.student_learning_style,
        conversation=guide.conversation,
    )
    guide.add_interaction(user_message=message, assistant_message=response)

    await guide_repository.save(guide=guide)

    return CreateMessageResponse(message=response.parts[0].content)


@router.delete("/{guide_id}")
async def delete_guide(
    guide_id: str,
    guide_repository: IGuideRepository = Depends(GuideRepository()),
) -> None:
    await guide_repository.delete(id=guide_id)
