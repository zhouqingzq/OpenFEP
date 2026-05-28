/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_M16_AUTH_TOKEN?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
