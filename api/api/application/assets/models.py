from pydantic import BaseModel, Field

from api.domain.aggregates.asset import Concept


class CreateAssetRequest(BaseModel):
    name: str = Field(description="The name of the file being uploaded, without the extension.")
    content: str = Field(description="The Markdown content of the file being uploaded.")


class CreateAssetResponse(BaseModel):
    id: str = Field(description="The ID of the asset that was created.")


class AssetOverview(BaseModel):
    id: str = Field(description="The ID of the asset.")
    name: str = Field(description="The name of the file, without the extension.")
    concepts: list[Concept] = Field(description="The concepts extracted from the asset.")


class ListAssetsResponse(BaseModel):
    assets: list[AssetOverview] = Field(description="A list of assets, without their content.")
