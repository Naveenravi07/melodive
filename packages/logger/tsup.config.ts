import { defineConfig } from "tsup";

export default defineConfig({
  watch:false,
  entry: ["src/index.ts"],
  clean: true,
  format: ["cjs", "esm"],
  dts: true,
  outDir: "dist",
  outExtension({ format }) {
    return {
      js: format === "esm" ? ".mjs" : ".cjs" 
    };
  },
  splitting: false,
  shims: false,
});
