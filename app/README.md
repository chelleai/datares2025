# Chelle Web App

This is the primary web app for Chelle users. Here, instructors can upload content to inform the Knowledge Model, and students can interact with the Knowledge Model to build and reinforce understanding. Instructors can also construct course materials and assess student work.

## Tech Stack

### Tanstack Router

We use Tanstack Router for type-safe routing with built-in data loading. The router integrates seamlessly with our OpenAPI-generated client to provide a fully type-safe data fetching experience. This means you get autocomplete for API endpoints and type checking for response data.

The router is configured with a context that provides access to our API clients throughout the application. This context is available in all routes and components:

```typescript
export interface AppContext {
  api: {
    assets: {
      client: Client<assetsPaths>;
      query: OpenapiQueryClient<assetsPaths>;
    };
  };
}
```

Routes are defined using `createFileRoute` and can include loaders for data fetching. Loaders are the recommended way to fetch initial data for a route. They run before the route is rendered and can access the API client through the route context. This pattern ensures data is available immediately when the route renders:

```typescript
export const Route = createFileRoute("/your-route")({
  loader: async ({ context: { api } }) => {
    // Fetch initial data
    const data = await api.assets.client.GET("/rest/assets/endpoint");
    return data?.data ?? [];
  },
  component: YourComponent,
});
```

### Tanstack Query

We leverage Tanstack Query through our OpenAPI-generated query client for efficient data fetching, caching, and state management. The integration provides type-safe access to all API endpoints while maintaining the powerful features of Tanstack Query.

#### Using Queries

Queries are the primary way to fetch and cache data in your components. They work in conjunction with route loaders to provide a smooth data loading experience. Here's a detailed breakdown of how to use queries:

```typescript
function YourComponent() {
  // Get initial data from the route loader - this data is available immediately
  const initialData = Route.useLoaderData();
  const { api } = Route.useRouteContext();

  // Use the query with options
  const { data, isLoading } = api.assets.query.useQuery(
    "get",
    "/rest/assets/endpoint",
    {
      initialData, // Pre-populate with loader data to prevent loading flicker
    },
    {
      // Additional TanStack Query options
      refetchInterval: 5000, // Automatically refresh data every 5 seconds
      staleTime: 1000 * 60, // Keep data fresh for 1 minute before refetching
    }
  );

  if (isLoading) return <Loading />;
  return <YourUI data={data} />;
}
```

The query above demonstrates several important patterns:

1. Using loader data as `initialData` to prevent loading states on first render
2. Setting a `refetchInterval` for real-time updates
3. Configuring `staleTime` to optimize refetching behavior
4. Proper handling of loading states

#### Using Mutations

Mutations are used to create, update, or delete data. They work hand-in-hand with queries to ensure your UI stays in sync with the server state. Here's how to effectively use mutations:

```typescript
function YourComponent() {
  const { api } = Route.useRouteContext();

  const mutation = api.assets.query.useMutation(
    "post",
    "/rest/assets/endpoint",
    {
      // Mutation options
      onSuccess: () => {
        // Invalidate and refetch queries after successful mutation
        api.assets.query.invalidateQueries(["queryKey"]);
      },
    },
  );

  const handleSubmit = (data) => {
    mutation.mutate({ body: data });
  };
}
```

Key mutation features:

- Type-safe mutation methods (`post`, `put`, `patch`, `delete`)
- Automatic error handling and loading states
- Cache invalidation through `invalidateQueries`
- Optimistic updates (can be added through `onMutate`)

#### Advanced Query Options

Our setup supports sophisticated data fetching patterns. Here are some advanced use cases:

```typescript
// Conditional refetching - useful for polling based on data state
const { data } = api.assets.query.useQuery(
  "get",
  "/rest/assets/endpoint",
  {},
  {
    refetchInterval: (query) => {
      // Refetch every 3 seconds if certain condition is met
      return query.state.data?.someCondition ? 3000 : false;
    },
  },
);

// Parallel queries - fetch multiple resources simultaneously
const [queryA, queryB] = api.assets.query.useQueries([
  {
    queryKey: ["a"],
    path: "/rest/assets/endpoint-a",
  },
  {
    queryKey: ["b"],
    path: "/rest/assets/endpoint-b",
  },
]);
```

These patterns enable:

- Dynamic polling based on data state
- Efficient parallel data fetching
- Optimized network requests
- Consistent loading and error states

### Project Structure

Our project follows a clear separation of concerns to maintain scalability and developer ergonomics:

```
src/
├── components/        # Reusable UI components
│   ├── ui/           # ShadCN components
│   └── feature/      # Feature-specific components
├── pages/            # Page components
│   └── integrations/ # Feature pages
└── routes/           # Route definitions and data loading
```

This structure provides several benefits:

- Clear separation between routing logic and UI components
- Centralized data fetching in route files
- Reusable UI components through the component library
- Feature-specific components kept close to their usage

### ShadCN Components

We use ShadCN as our component library for its excellent combination of flexibility and consistency. These components are built on top of Radix UI primitives and styled with Tailwind CSS, providing accessible and customizable UI elements.

Here's a practical example showing how to compose ShadCN components:

```typescript
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";

function YourComponent() {
  return (
    <Card>
      <Card.Header>
        <Card.Title>Your Title</Card.Title>
        <Card.Description>Your description</Card.Description>
      </Card.Header>
      <Card.Content>
        <div className="space-y-4">
          <Input placeholder="Enter something..." />
          <Button variant="default">Submit</Button>
        </div>
      </Card.Content>
    </Card>
  );
}
```

Key benefits of using ShadCN:

- Consistent styling across the application
- Built-in accessibility features
- Easy customization through Tailwind CSS
- Composable component API

#### Best Practices

1. **Data Loading**:
   Data loading should be predictable and efficient. Follow these guidelines:

   - Use route loaders for initial data fetch to avoid loading states on first render
   - Provide `initialData` to queries from your route loader
   - Set appropriate `staleTime` and `cacheTime` based on your data's freshness requirements
   - Consider implementing optimistic updates for mutations

2. **Component Structure**:
   Keep your codebase maintainable by following these patterns:

   - Route components should focus on data fetching and state management
   - Move UI logic to page components to separate concerns
   - Use ShadCN components as building blocks for consistent UI
   - Create feature-specific components when needed

3. **Error Handling**:
   Robust error handling is crucial for a good user experience:

   - Always handle loading and error states appropriately
   - Use the built-in error boundaries for route-level errors
   - Provide meaningful error messages to users
   - Consider implementing retry logic for failed requests

4. **Performance**:
   Optimize performance through smart data management:
   - Use the `enabled` option to control when queries run
   - Implement proper cache invalidation strategies
   - Set up background refetching only when necessary
   - Consider using suspense mode for loading states
