import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  use: { baseURL: "http://127.0.0.1:3000", ...devices["Desktop Chrome"] },
  webServer: { command: "npm run dev", url: "http://127.0.0.1:3000", reuseExistingServer: true },
});
