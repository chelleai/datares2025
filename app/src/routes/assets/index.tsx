import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { useAssets, useDeleteAsset } from "@/lib/hooks";
import { createFileRoute, Link } from "@tanstack/react-router";
import { AlertTriangle, PlusCircle, Trash2 } from "lucide-react";

export const Route = createFileRoute("/assets/")({
  component: AssetsIndexPage,
});

function AssetsIndexPage() {
  const { data, isLoading, isError } = useAssets();
  const deleteAssetMutation = useDeleteAsset();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900 mx-auto"></div>
          <p className="mt-4">Loading assets...</p>
        </div>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <AlertTriangle className="h-12 w-12 text-red-500 mx-auto" />
          <p className="mt-4">Error loading assets. Please try again later.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto">
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl font-bold">Assets</h1>
        <Button asChild>
          <Link to="/assets/new">
            <PlusCircle className="mr-2 h-4 w-4" />
            Upload New Asset
          </Link>
        </Button>
      </div>

      {data?.assets.length === 0 ? (
        <div className="text-center p-12 border rounded-md">
          <p className="text-xl mb-4">No assets found</p>
          <p className="text-gray-500 mb-8">
            Start by uploading a Markdown document to extract concepts.
          </p>
          <Button asChild>
            <Link to="/assets/new">
              <PlusCircle className="mr-2 h-4 w-4" />
              Upload New Asset
            </Link>
          </Button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {data.assets.map((asset) => (
            <Card key={asset.id}>
              <CardHeader>
                <CardTitle className="truncate" title={asset.name}>
                  {asset.name}
                </CardTitle>
                <CardDescription>
                  {asset.concepts.length} concepts extracted
                </CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-gray-500 mb-2">Concepts:</p>
                <div className="flex flex-wrap gap-2">
                  {asset.concepts.slice(0, 5).map((concept) => (
                    <span
                      key={concept.term}
                      className="px-2 py-1 bg-gray-100 text-xs rounded-md"
                    >
                      {concept.term}
                    </span>
                  ))}
                  {asset.concepts.length > 5 && (
                    <span className="px-2 py-1 bg-gray-100 text-xs rounded-md">
                      +{asset.concepts.length - 5} more
                    </span>
                  )}
                </div>
              </CardContent>
              <CardFooter className="flex justify-between">
                <Button variant="outline" asChild>
                  <Link to={`/assets/${asset.id}`}>View Details</Link>
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => deleteAssetMutation.mutate(asset.id)}
                  disabled={deleteAssetMutation.isPending}
                  title="Delete asset"
                >
                  <Trash2 className="h-4 w-4 text-red-500" />
                </Button>
              </CardFooter>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}