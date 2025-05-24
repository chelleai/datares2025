import { cn } from "@/lib/utils";
import { Link, useMatchRoute } from "@tanstack/react-router";
import { Button } from "./button";

interface SidebarNavProps extends React.HTMLAttributes<HTMLDivElement> {}

export function SidebarNav({ className, ...props }: SidebarNavProps) {
  return (
    <div className={cn("pb-12", className)} {...props}>
      <div className="space-y-4 py-4">
        <div className="px-3 py-2">
          <h2 className="mb-2 px-4 text-lg font-semibold tracking-tight">
            DataRes
          </h2>
          <div className="space-y-1">
            <NavLink to="/" label="Home" />
            <NavLink to="/assets" label="Assets" />
            <NavLink to="/guides" label="Guides" />
          </div>
        </div>
      </div>
    </div>
  );
}

interface NavLinkProps {
  to: string;
  label: string;
}

function NavLink({ to, label }: NavLinkProps) {
  const matchRoute = useMatchRoute();
  const isActive = matchRoute({ to });

  return (
    <Button
      asChild
      variant={isActive ? "secondary" : "ghost"}
      className={cn("w-full justify-start", {
        "bg-accent": isActive,
      })}
    >
      <Link to={to}>{label}</Link>
    </Button>
  );
}