import { expect, test } from "@playwright/test";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { strFromU8, unzipSync } from "fflate";

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
  await expect(
    page
      .getByTestId("overlays")
      .locator(".line-chart-wrapper, .bar-chart-wrapper"),
  ).toHaveCount(2);
  await expect(page.getByTestId("projections")).toContainText("baseline");
  await expect(page.getByTestId("projections")).toContainText("20 particles");
  await expect(page.getByTestId("cells").locator(".gen-cell")).toHaveCount(5);

  await page.locator(".gen-table tbody tr").nth(1).click();
  await expect(page.getByTestId("trajectories")).toContainText("Gen 1");
});

test("exports particles, trajectories, and every plot as one zip", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByLabel("Load run.jsonl").setInputFiles(fixture);
  await expect(page.getByTestId("status")).toHaveText("Finished");

  const charts = await page.locator("[data-export-name]").count();
  const downloading = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export outputs" }).click();
  const download = await downloading;
  expect(download.suggestedFilename()).toMatch(/^\d{8}-\d{6}-outputs\.zip$/);

  const entries = unzipSync(
    new Uint8Array(await readFile(await download.path())),
  );
  const names = Object.keys(entries);

  // 5 generations of 100 particles and 20 trajectories each in the fixture.
  const particles = strFromU8(entries["particles.csv"]!).split("\n");
  expect(particles[0]).toBe(
    "generation,particle,weight,distance,seed,r0,initial_infections",
  );
  expect(particles).toHaveLength(1 + 5 * 100);
  const trajectories = strFromU8(entries["trajectories.csv"]!).split("\n");
  expect(trajectories[0]).toBe("generation,particle,index,value");
  expect(new Set(trajectories.slice(1).map((r) => r.split(",")[0]))).toEqual(
    new Set(["0", "1", "2", "3", "4"]),
  );

  const plots = names.filter((n) => n.startsWith("plots/"));
  expect(charts).toBeGreaterThan(0);
  expect(plots).toHaveLength(charts);
  for (const name of plots) {
    expect(name).toMatch(/\.png$/);
    expect([...entries[name]!.slice(0, 4)]).toEqual([0x89, 0x50, 0x4e, 0x47]);
    expect(entries[name]!.length).toBeGreaterThan(1000);
  }
});
