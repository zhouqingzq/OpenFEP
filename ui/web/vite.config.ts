import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const rootDir = path.dirname(fileURLToPath(import.meta.url));
const clientRoot = path.resolve(rootDir, "../packages/consciousness-client/src");

export default defineConfig({
  resolve: {
    alias: {
      "@segments/consciousness-client": path.resolve(clientRoot, "index.ts"),
      "@segments/consciousness-client/validate": path.resolve(clientRoot, "validate.ts"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/health": {
        target: "http://127.0.0.1:8765",
        changeOrigin: true,
      },
      "/v1": {
        target: "http://127.0.0.1:8765",
        changeOrigin: true,
        ws: true,
      },
    },
  },
});
