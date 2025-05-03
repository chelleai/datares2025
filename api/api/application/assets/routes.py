from fastapi import APIRouter, Depends

from api.application.assets.models import AssetOverview, CreateAssetRequest, CreateAssetResponse, ListAssetsResponse
from api.domain.aggregates.asset.aggregate import Asset
from api.domain.interfaces.asset_processor import IAssetProcessor
from api.domain.repositories import IAssetRepository
from api.infrastructure.repositories import AssetRepository
from api.infrastructure.services.asset_processor import AssetProcessor

router = APIRouter(prefix="/assets", tags=["Assets"])


@router.get("", response_model=ListAssetsResponse)
async def list_assets(asset_repository: IAssetRepository = Depends(AssetRepository())) -> ListAssetsResponse:
    """List all assets.

    An asset is a Markdown document that has been processed by AI to extract concepts.
    These overviews show the name of the asset and the concepts it contains.

    A Concept is an educational term that has been extracted from the asset because it contains citations.
    These citations have been synthesized into a definition.
    """
    assets = await asset_repository.list()
    return ListAssetsResponse(assets=[AssetOverview.model_validate(asset.model_dump()) for asset in assets])


@router.post("", response_model=CreateAssetResponse)
async def create_asset(
    body: CreateAssetRequest,
    asset_repository: IAssetRepository = Depends(AssetRepository()),
    asset_processor: IAssetProcessor = Depends(AssetProcessor()),
) -> CreateAssetResponse:
    """Upload Markdown content to be converted into an Asset.

    This process analyzes the Markdown content for concepts.
    """
    asset = Asset.create(name=body.name, content=body.content)
    concepts = await asset_processor.process(content=body.content)

    for concept in concepts:
        asset.add_concept(term=concept.term, citations=concept.citations, definition=concept.definition)

    await asset_repository.save(asset=asset)
    return CreateAssetResponse(id=asset.id)


@router.get("/{id}", response_model=Asset)
async def get_asset(id: str, asset_repository: IAssetRepository = Depends(AssetRepository())) -> Asset:
    """Get all details about a particular Asset.

    This includes its name, Markdown content, and all concepts extracted from it.
    """
    asset = await asset_repository.get(id=id)
    return asset


@router.delete("/{id}")
async def delete_asset(id: str, asset_repository: IAssetRepository = Depends(AssetRepository())) -> None:
    """Delete an Asset."""
    await asset_repository.delete(id=id)
