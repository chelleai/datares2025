import { createFileRoute } from "@tanstack/react-router";
import { Outlet } from "@tanstack/react-router";

export const Route = createFileRoute("/assets")({
  component: AssetsLayout,
});

function AssetsLayout() {
  return <Outlet />;
}