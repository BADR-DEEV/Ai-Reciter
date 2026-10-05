// Runs once when the Next server starts. Node-only work lives in instrumentation-node.ts so the
// Edge bundle never sees node: imports (Next only keeps this branch in the Node.js build).
export async function register() {
  if (process.env.NEXT_RUNTIME === "nodejs") await import("./instrumentation-node");
}
