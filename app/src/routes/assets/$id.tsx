import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useAsset } from "@/lib/hooks";
import { createFileRoute, Link, useParams } from "@tanstack/react-router";
import { AlertTriangle, ArrowLeft } from "lucide-react";
import ReactMarkdown from "react-markdown";

export const Route = createFileRoute("/assets/$id")({
  component: AssetDetailPage,
});

function AssetDetailPage() {
  const { id } = useParams({ from: "/assets/$id" });
  const { data: asset, isLoading, isError } = useAsset(id);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900 mx-auto"></div>
          <p className="mt-4">Loading asset details...</p>
        </div>
      </div>
    );
  }

  if (isError || !asset) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <AlertTriangle className="h-12 w-12 text-red-500 mx-auto" />
          <p className="mt-4">Error loading asset details. Please try again later.</p>
          <Button variant="outline" asChild className="mt-4">
            <Link to="/assets">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Assets
            </Link>
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto">
      <Button variant="outline" asChild className="mb-6">
        <Link to="/assets">
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to Assets
        </Link>
      </Button>

      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-4">{asset.name}</h1>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <Card className="p-6">
            <h2 className="text-xl font-semibold mb-4">Content</h2>
            <div className="prose prose-sm max-w-none">
              <ReactMarkdown>{asset.content}</ReactMarkdown>
            </div>
          </Card>
        </div>

        <div>
          <Card className="p-6">
            <h2 className="text-xl font-semibold mb-4">Extracted Concepts</h2>
            {asset.concepts.length === 0 ? (
              <p className="text-gray-500">No concepts were extracted from this asset.</p>
            ) : (
              <div className="space-y-4">
                {asset.concepts.map((concept) => (
                  <div key={concept.term} className="border-b pb-4 last:border-b-0 last:pb-0">
                    <h3 className="font-medium text-lg">{concept.term}</h3>
                    <p className="text-sm text-gray-700 mt-1">{concept.definition}</p>
                    <div className="mt-2">
                      <p className="text-xs text-gray-500 mb-1">Citations:</p>
                      <ul className="list-disc list-inside text-xs text-gray-500">
                        {concept.citations.map((citation, index) => (
                          <li key={index} className="truncate" title={citation}>
                            {citation}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}