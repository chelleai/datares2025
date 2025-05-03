from typing import Self

from tinydb import Query, where

from api.domain.aggregates.asset import Asset
from api.domain.aggregates.guide import Guide
from api.domain.repositories import IAssetRepository, IGuideRepository
from api.infrastructure.db import db


class AssetRepository(IAssetRepository):
    def __call__(self) -> Self:
        return self  # trick to help with validation against the interface

    async def get(self, *, id: str) -> Asset:
        query = Query()
        matches = db.search(query.id == id)
        if len(matches) != 1:
            raise ValueError("Asset not found")

        asset = matches[0]

        return Asset.model_validate(asset)

    async def list(self) -> list[Asset]:
        query = Query()
        assets = db.search(query.table == "asset")
        return [Asset.model_validate(asset) for asset in assets]

    async def save(self, *, asset: Asset) -> None:
        existing_asset = db.search(Query().id == asset.id)
        if len(existing_asset) > 0:
            db.update({**asset.model_dump()}, where("id") == asset.id)
        else:
            db.insert({**asset.model_dump(), "table": "asset"})

    async def delete(self, *, id: str) -> None:
        query = Query()
        db.remove(query.id == id)


class GuideRepository(IGuideRepository):
    def __call__(self) -> Self:
        return self  # trick to help with validation against the interface

    async def get(self, *, id: str) -> Guide:
        query = Query()
        matches = db.search(query.id == id)
        if len(matches) != 1:
            raise ValueError("Guide not found")

        guide = matches[0]

        return Guide.model_validate(guide)

    async def list(self) -> list[Guide]:
        query = Query()
        guides = db.search(query.table == "guide")
        return [Guide.model_validate(guide) for guide in guides]

    async def save(self, *, guide: Guide) -> None:
        existing_guide = db.search(Query().id == guide.id)
        if len(existing_guide) > 0:
            db.update({**guide.model_dump()}, where("id") == guide.id)
        else:
            db.insert({**guide.model_dump(), "table": "guide"})

    async def delete(self, *, id: str) -> None:
        query = Query()
        db.remove(query.id == id)
