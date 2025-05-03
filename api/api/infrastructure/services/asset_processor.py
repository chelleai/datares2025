from typing import Self

from api.domain.interfaces.asset_processor import IAssetProcessor, IdentifiedConcept


class AssetProcessor(IAssetProcessor):
    def __call__(self) -> Self:
        return self  # to help with validation against the interface

    async def process(self, *, content: str) -> list[IdentifiedConcept]:
        return [
            IdentifiedConcept(
                term="example",
                citations=["citation1", "citation2"],
                definition="definition",
            )
        ]
