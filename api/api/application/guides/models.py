from pydantic import BaseModel, Field

from api.domain.aggregates.guide import GuideConcept


class CreateGuideRequest(BaseModel):
    name: str = Field(description="The name of the guide.")
    concepts: list[GuideConcept] = Field(description="The concepts that the guide will teach.")
    student_learning_style: str = Field(description="The learning style of the student.")


class CreateMessageRequest(BaseModel):
    message: str = Field(description="The user's message, text only.")


class CreateGuideResponse(BaseModel):
    id: str = Field(description="The ID of the guide.")


class GuideOverview(BaseModel):
    id: str = Field(description="The ID of the guide.")
    name: str = Field(description="The name of the guide.")
    concepts: list[GuideConcept] = Field(description="The concepts that the guide will teach.")


class ListGuidesResponse(BaseModel):
    guides: list[GuideOverview] = Field(description="The guides that the user has created.")


class CreateMessageResponse(BaseModel):
    message: str = Field(description="The assistant's message.")
