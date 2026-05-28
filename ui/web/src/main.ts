import { ConsciousnessWebApp, loadSessionConfig } from "./app.js";

const root = document.querySelector("#app") as HTMLElement | null;
if (!root) {
  throw new Error("#app root missing");
}

const app = new ConsciousnessWebApp(loadSessionConfig());
app.mount(root);
