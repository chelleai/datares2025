import { AlertCircle } from "lucide-react";

export function ErrorComponent({
  title = "Error",
  error,
  className,
  fullScreen = false,
}: {
  title?: string;
  error: Error | string | unknown;
  className?: string;
  fullScreen?: boolean;
}) {
  let message = "";
  if (error instanceof Error) {
    message = error.message;
  } else if (typeof error === "string") {
    message = error;
  } else {
    message = "An unknown error occurred";
  }

  const containerClasses = fullScreen
    ? "fixed inset-0 flex items-center justify-center bg-black/5 backdrop-blur-sm z-50"
    : "";

  return (
    <div className={containerClasses}>
      <div
        className={`bg-red-50 border-l-4 border-red-500 p-4 rounded-r ${className}`}
      >
        <div className="flex">
          <AlertCircle className="h-6 w-6 text-red-500" />
          <div className="ml-3">
            <p className="text-red-700 font-medium">{title}</p>
            <p className="text-red-600 mt-1">{message}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
