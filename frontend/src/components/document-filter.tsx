import { Search } from "lucide-react";

export function DocumentFilter() {
  return (
    <div className="flex min-h-10 items-center gap-2 rounded-md border border-input bg-muted/20 px-3 text-sm text-muted-foreground">
      <Search className="h-4 w-4 shrink-0" aria-hidden="true" />
      <span className="truncate">Document filter arrives with API integration</span>
    </div>
  );
}
