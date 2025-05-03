import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { useDeleteGuide, useGuides } from "@/lib/hooks";
import { createFileRoute, Link } from "@tanstack/react-router";
import { AlertTriangle, MessageCircle, PlusCircle, Trash2 } from "lucide-react";

export const Route = createFileRoute("/guides/")({
  component: GuidesIndexPage,
});

function GuidesIndexPage() {
  const { data, isLoading, isError } = useGuides();
  const deleteGuideMutation = useDeleteGuide();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900 mx-auto"></div>
          <p className="mt-4">Loading guides...</p>
        </div>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <AlertTriangle className="h-12 w-12 text-red-500 mx-auto" />
          <p className="mt-4">Error loading guides. Please try again later.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto">
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl font-bold">Guides</h1>
        <Button asChild>
          <Link to="/guides/new">
            <PlusCircle className="mr-2 h-4 w-4" />
            Create New Guide
          </Link>
        </Button>
      </div>

      {data?.guides.length === 0 ? (
        <div className="text-center p-12 border rounded-md">
          <p className="text-xl mb-4">No guides found</p>
          <p className="text-gray-500 mb-8">
            Create a guide to start learning with an AI tutor.
          </p>
          <Button asChild>
            <Link to="/guides/new">
              <PlusCircle className="mr-2 h-4 w-4" />
              Create New Guide
            </Link>
          </Button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {data.guides.map((guide) => (
            <Card key={guide.id}>
              <CardHeader>
                <CardTitle className="truncate" title={guide.name}>
                  {guide.name}
                </CardTitle>
                <CardDescription>
                  {guide.concepts.length} concepts included
                </CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-gray-500 mb-2">Concepts:</p>
                <div className="flex flex-wrap gap-2">
                  {guide.concepts.slice(0, 5).map((concept) => (
                    <span
                      key={concept.term}
                      className="px-2 py-1 bg-gray-100 text-xs rounded-md"
                    >
                      {concept.term}
                    </span>
                  ))}
                  {guide.concepts.length > 5 && (
                    <span className="px-2 py-1 bg-gray-100 text-xs rounded-md">
                      +{guide.concepts.length - 5} more
                    </span>
                  )}
                </div>
              </CardContent>
              <CardFooter className="flex justify-between">
                <Button variant="default" asChild>
                  <Link to={`/guides/${guide.id}`}>
                    <MessageCircle className="mr-2 h-4 w-4" />
                    Chat
                  </Link>
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => deleteGuideMutation.mutate(guide.id)}
                  disabled={deleteGuideMutation.isPending}
                  title="Delete guide"
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