import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { useCreateAsset } from "@/lib/hooks";
import { zodResolver } from "@hookform/resolvers/zod";
import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { ArrowLeft, Upload } from "lucide-react";
import { useForm } from "react-hook-form";
import { z } from "zod";

export const Route = createFileRoute("/assets/new")({
  component: NewAssetPage,
});

const formSchema = z.object({
  name: z.string().min(1, "Name is required"),
  content: z.string().min(1, "Content is required"),
});

type FormValues = z.infer<typeof formSchema>;

function NewAssetPage() {
  const navigate = useNavigate();
  const createAssetMutation = useCreateAsset();

  const form = useForm<FormValues>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      name: "",
      content: "",
    },
  });

  const onSubmit = async (values: FormValues) => {
    try {
      await createAssetMutation.mutateAsync(values);
      navigate({ to: "/assets" });
    } catch (error) {
      console.error("Error creating asset:", error);
    }
  };

  return (
    <div className="container mx-auto">
      <Button variant="outline" asChild className="mb-6">
        <Link to="/assets">
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to Assets
        </Link>
      </Button>

      <div className="mb-8">
        <h1 className="text-3xl font-bold">Upload New Asset</h1>
        <p className="text-gray-500 mt-2">
          Upload a Markdown document to extract concepts and their definitions.
        </p>
      </div>

      <Card>
        <CardContent className="pt-6">
          <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
              <FormField
                control={form.control}
                name="name"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Asset Name</FormLabel>
                    <FormControl>
                      <Input placeholder="Enter asset name" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="content"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Markdown Content</FormLabel>
                    <FormControl>
                      <Textarea
                        placeholder="# Your Markdown Content Here"
                        className="min-h-[300px] font-mono"
                        {...field}
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <div className="flex justify-end">
                <Button
                  type="submit"
                  disabled={createAssetMutation.isPending}
                  className="min-w-[120px]"
                >
                  {createAssetMutation.isPending ? (
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                  ) : (
                    <>
                      <Upload className="mr-2 h-4 w-4" /> Upload
                    </>
                  )}
                </Button>
              </div>
            </form>
          </Form>
        </CardContent>
      </Card>
    </div>
  );
}