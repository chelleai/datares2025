from typing import Self

from tinydb import Query

from api.domain.aggregates.asset.aggregate import Asset
from api.domain.repositories import IAssetRepository
from api.infrastructure.db import db


class AssetRepository(IAssetRepository):
    def __call__(self) -> Self:
        return self  # to help with validation against the interface

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
        db.insert({**asset.model_dump(), "table": "asset"})

    async def delete(self, *, id: str) -> None:
        query = Query()
        db.remove(query.id == id)
