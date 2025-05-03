import { createRootRouteWithContext, Outlet } from "@tanstack/react-router";

export interface AppContext {}

export const Route = createRootRouteWithContext<AppContext>()({
  component: Layout,
});

function Layout() {
  return (
    <main className="flex flex-1 flex-col">
      <div className="flex-1 overflow-y-auto">
        <Outlet />
      </div>
    </main>
  );
}
