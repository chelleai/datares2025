import { createFileRoute } from "@tanstack/react-router";
import { Outlet } from "@tanstack/react-router";

export const Route = createFileRoute("/guides")({
  component: GuidesLayout,
});

function GuidesLayout() {
  return <Outlet />;
}