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
import { GuideConcept, useCreateGuide } from "@/lib/hooks";
import { zodResolver } from "@hookform/resolvers/zod";
import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { ArrowLeft, Plus, Save, Trash2 } from "lucide-react";
import { useState } from "react";
import { useFieldArray, useForm } from "react-hook-form";
import { z } from "zod";

export const Route = createFileRoute("/guides/new")({
  component: NewGuidePage,
});

const conceptSchema = z.object({
  term: z.string().min(1, "Term is required"),
  definition: z.string().min(1, "Definition is required"),
});

const formSchema = z.object({
  name: z.string().min(1, "Name is required"),
  student_learning_style: z.string().min(1, "Learning style is required"),
  concepts: z.array(conceptSchema).min(1, "At least one concept is required"),
});

type FormValues = z.infer<typeof formSchema>;

function NewGuidePage() {
  const navigate = useNavigate();
  const createGuideMutation = useCreateGuide();

  const form = useForm<FormValues>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      name: "",
      student_learning_style: "",
      concepts: [{ term: "", definition: "" }],
    },
  });

  const { fields, append, remove } = useFieldArray({
    control: form.control,
    name: "concepts",
  });

  const onSubmit = async (values: FormValues) => {
    try {
      await createGuideMutation.mutateAsync(values);
      navigate({ to: "/guides" });
    } catch (error) {
      console.error("Error creating guide:", error);
    }
  };

  return (
    <div className="container mx-auto">
      <Button variant="outline" asChild className="mb-6">
        <Link to="/guides">
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to Guides
        </Link>
      </Button>

      <div className="mb-8">
        <h1 className="text-3xl font-bold">Create New Guide</h1>
        <p className="text-gray-500 mt-2">
          Create an AI tutor specialized in specific concepts to help you learn.
        </p>
      </div>

      <Card>
        <CardContent className="pt-6">
          <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <FormField
                  control={form.control}
                  name="name"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Guide Name</FormLabel>
                      <FormControl>
                        <Input placeholder="e.g., JavaScript Fundamentals" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="student_learning_style"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Your Learning Style</FormLabel>
                      <FormControl>
                        <Input 
                          placeholder="e.g., Visual learner, prefer examples" 
                          {...field} 
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>

              <div>
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-medium">Concepts</h3>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => append({ term: "", definition: "" })}
                  >
                    <Plus className="mr-1 h-4 w-4" />
                    Add Concept
                  </Button>
                </div>

                {fields.length === 0 && (
                  <p className="text-red-500 text-sm mb-4">
                    At least one concept is required.
                  </p>
                )}

                <div className="space-y-4">
                  {fields.map((field, index) => (
                    <div
                      key={field.id}
                      className="p-4 border rounded-md grid gap-4"
                    >
                      <div className="flex justify-between">
                        <h4 className="font-medium">Concept {index + 1}</h4>
                        {fields.length > 1 && (
                          <Button
                            type="button"
                            variant="ghost"
                            size="icon"
                            onClick={() => remove(index)}
                          >
                            <Trash2 className="h-4 w-4 text-red-500" />
                          </Button>
                        )}
                      </div>

                      <FormField
                        control={form.control}
                        name={`concepts.${index}.term`}
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Term</FormLabel>
                            <FormControl>
                              <Input placeholder="e.g., Variables" {...field} />
                            </FormControl>
                            <FormMessage />
                          </FormItem>
                        )}
                      />

                      <FormField
                        control={form.control}
                        name={`concepts.${index}.definition`}
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Definition</FormLabel>
                            <FormControl>
                              <Textarea
                                placeholder="e.g., A variable is a container for storing data values..."
                                className="min-h-[100px]"
                                {...field}
                              />
                            </FormControl>
                            <FormMessage />
                          </FormItem>
                        )}
                      />
                    </div>
                  ))}
                </div>
              </div>

              <div className="flex justify-end">
                <Button
                  type="submit"
                  disabled={createGuideMutation.isPending}
                  className="min-w-[120px]"
                >
                  {createGuideMutation.isPending ? (
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                  ) : (
                    <>
                      <Save className="mr-2 h-4 w-4" /> Create Guide
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