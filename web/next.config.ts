import type { NextConfig } from "next";
import { PHASE_DEVELOPMENT_SERVER } from "next/constants";

// A dev server must never invalidate a presentation server's compiled assets.
const config = (phase: string): NextConfig => ({
  poweredByHeader: false, reactStrictMode: true, devIndicators: false,
  distDir: phase === PHASE_DEVELOPMENT_SERVER ? ".next-dev" : ".next-production",
});
export default config;
