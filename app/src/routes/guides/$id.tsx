import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { useGuide, useSendMessage } from "@/lib/hooks";
import { createFileRoute, Link, useParams } from "@tanstack/react-router";
import { AlertTriangle, ArrowLeft, Send } from "lucide-react";
import { useState } from "react";

export const Route = createFileRoute("/guides/$id")({
  component: GuideChatPage,
});

function GuideChatPage() {
  const { id } = useParams({ from: "/guides/$id" });
  const { data: guide, isLoading, isError } = useGuide(id);
  const [message, setMessage] = useState("");
  const sendMessageMutation = useSendMessage(id);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!message.trim()) return;

    try {
      await sendMessageMutation.mutateAsync({ message });
      setMessage("");
    } catch (error) {
      console.error("Error sending message:", error);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900 mx-auto"></div>
          <p className="mt-4">Loading guide...</p>
        </div>
      </div>
    );
  }

  if (isError || !guide) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <AlertTriangle className="h-12 w-12 text-red-500 mx-auto" />
          <p className="mt-4">Error loading guide. Please try again later.</p>
          <Button variant="outline" asChild className="mt-4">
            <Link to="/guides">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Guides
            </Link>
          </Button>
        </div>
      </div>
    );
  }

  const messages = [...guide.user_messages, ...guide.assistant_messages].sort(
    (a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
  );

  return (
    <div className="container max-w-5xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-4">
          <Button variant="outline" asChild>
            <Link to="/guides">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back
            </Link>
          </Button>
          <h1 className="text-2xl font-bold">{guide.name}</h1>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="lg:col-span-1">
          <Card className="p-4">
            <h2 className="text-lg font-semibold mb-3">Guide Concepts</h2>
            <Separator className="mb-3" />
            <div className="space-y-3">
              {guide.concepts.map((concept) => (
                <div key={concept.term} className="space-y-1">
                  <p className="font-medium">{concept.term}</p>
                  <p className="text-sm text-gray-600">{concept.definition}</p>
                </div>
              ))}
            </div>
            <Separator className="my-3" />
            <div>
              <p className="text-sm font-medium">Learning Style:</p>
              <p className="text-sm text-gray-600">{guide.student_learning_style}</p>
            </div>
          </Card>
        </div>

        <div className="lg:col-span-3 flex flex-col h-[calc(100vh-180px)]">
          <Card className="flex-1 flex flex-col overflow-hidden p-0">
            <ScrollArea className="flex-1 p-4">
              {messages.length === 0 ? (
                <div className="h-full flex items-center justify-center text-center p-8">
                  <div>
                    <h3 className="text-lg font-medium mb-2">Start a conversation</h3>
                    <p className="text-gray-600">
                      Ask questions about {guide.concepts.map(c => c.term).join(", ")}
                    </p>
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  {messages.map((msg, idx) => {
                    const isUser = msg.role === "user";
                    return (
                      <div
                        key={idx}
                        className={`flex ${isUser ? "justify-end" : "justify-start"}`}
                      >
                        <div
                          className={`max-w-[80%] rounded-lg p-3 ${
                            isUser
                              ? "bg-primary text-primary-foreground"
                              : "bg-muted"
                          }`}
                        >
                          <p className="whitespace-pre-wrap">{msg.parts[0].content}</p>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </ScrollArea>
            <div className="border-t p-4">
              <form onSubmit={handleSendMessage} className="flex gap-2">
                <Input
                  placeholder="Type your message..."
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  disabled={sendMessageMutation.isPending}
                  className="flex-1"
                />
                <Button
                  type="submit"
                  disabled={!message.trim() || sendMessageMutation.isPending}
                >
                  {sendMessageMutation.isPending ? (
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                  ) : (
                    <Send className="h-4 w-4" />
                  )}
                </Button>
              </form>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}