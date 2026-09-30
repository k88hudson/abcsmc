import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL: "http://localhost:5187",
    trace: "retain-on-failure",
  },
  webServer: {
    command: "pnpm exec vite --port 5187 --strictPort",
    url: "http://localhost:5187",
    reuseExistingServer: !process.env.CI,
  },
  projects: [{ name: "chromium", use: { browserName: "chromium" } }],
});
