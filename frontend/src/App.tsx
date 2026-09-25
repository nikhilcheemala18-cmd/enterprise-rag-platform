import { AppShell } from "@/components/app-shell";
import { LandingPage } from "@/components/landing-page";

export default function App() {
  const path = window.location.pathname;

  if (path === "/workspace" || path.startsWith("/workspace/")) {
    return <AppShell />;
  }

  return <LandingPage />;
}
