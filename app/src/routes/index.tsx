import { createFileRoute } from "@tanstack/react-router";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Link } from "@tanstack/react-router";

export const Route = createFileRoute("/")({
  component: HomePage,
});

function HomePage() {
  return (
    <div className="container mx-auto">
      <h1 className="text-3xl font-bold mb-8">Welcome to DataRes</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Assets</CardTitle>
            <CardDescription>
              Upload Markdown documents to extract concepts
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p>
              Assets are Markdown documents processed by AI to extract concepts and their definitions.
              Upload your educational content and see what concepts are identified.
            </p>
          </CardContent>
          <CardFooter>
            <Button asChild>
              <Link to="/assets">View Assets</Link>
            </Button>
          </CardFooter>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Guides</CardTitle>
            <CardDescription>
              Create conversational tutors for specific concepts
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p>
              Guides are AI tutors designed to help you learn specific concepts.
              Create a guide with concepts you want to learn about and start chatting.
            </p>
          </CardContent>
          <CardFooter>
            <Button asChild>
              <Link to="/guides">View Guides</Link>
            </Button>
          </CardFooter>
        </Card>
      </div>
    </div>
  );
}