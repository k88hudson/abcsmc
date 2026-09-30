import { expect, test } from "@playwright/test";
import { fileURLToPath } from "node:url";

const fixture = fileURLToPath(
  new URL("./fixtures/renewal.jsonl", import.meta.url),
);

test("loads a finished run from a file and renders every generation", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.getByTestId("status")).toHaveText("No run loaded");

  await page.getByLabel("Load run.jsonl").setInputFiles(fixture);

  await expect(page.getByTestId("status")).toHaveText("Finished");
  await expect(page.getByTestId("run-id")).toHaveText(/^\d{8}-\d{6}$/);
  await expect(page.locator(".gen-table tbody tr")).toHaveCount(5);
  await expect(page.locator(".gen-table tbody tr").first()).toContainText("∞");

  await expect(page.getByTestId("live")).toHaveCount(0);
  await expect(page.getByTestId("trajectories")).toContainText("Gen 4");
  await expect(page.getByTestId("overlays").locator("svg")).toHaveCount(2);
  await expect(page.getByTestId("cells").locator(".gen-cell")).toHaveCount(5);

  await page.locator(".gen-table tbody tr").nth(1).click();
  await expect(page.getByTestId("trajectories")).toContainText("Gen 1");
});
