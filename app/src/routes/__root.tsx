import { SidebarNav } from "@/components/ui/sidebar-nav";
import { createRootRouteWithContext, Link, Outlet } from "@tanstack/react-router";

export interface AppContext {}

export const Route = createRootRouteWithContext<AppContext>()({
  component: Layout,
});

function Layout() {
  return (
    <div className="flex h-screen">
      <div className="w-64 border-r">
        <SidebarNav />
      </div>
      <main className="flex flex-1 flex-col">
        <div className="flex-1 overflow-y-auto p-6">
          <Outlet />
        </div>
      </main>
    </div>
  );
}