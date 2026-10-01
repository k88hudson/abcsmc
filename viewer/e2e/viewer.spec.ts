import { expect, test } from "@playwright/test";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { strFromU8, unzipSync } from "fflate";

const fixture = fileURLToPath(
  new URL("./fixtures/renewal.jsonl", import.meta.url),
);

// The file input is the fallback for browsers without file handles, and the
// one way to hand the page a file from here, so most tests run without them.
test.describe("without file handles", () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => {
      delete (window as { showOpenFilePicker?: unknown }).showOpenFilePicker;
    });
  });

  test("loads a finished run from a file and renders every generation", async ({
    page,
  }) => {
    await page.goto("/");
    await expect(page.getByTestId("status")).toHaveText("No run loaded");

    await page.getByLabel("Load run.jsonl").setInputFiles(fixture);

    await expect(page.getByTestId("status")).toHaveText("Finished");
    await expect(page.getByTestId("run-id")).toHaveText(/^\d{8}-\d{6}$/);
    await expect(page.locator(".gen-table tbody tr")).toHaveCount(5);
    await expect(page.locator(".gen-table tbody tr").first()).toContainText(
      "∞",
    );

    await expect(page.getByTestId("live")).toHaveCount(0);
    // Target data sits on its own tab, as a chart and a table.
    const target = page.getByTestId("target");
    await expect(target).toBeHidden();
    await page.getByRole("tab", { name: "Target data" }).click();
    await expect(target).toContainText("42 observed points");
    await expect(target.locator(".line-chart-wrapper")).toHaveCount(1);
    await expect(target.locator("tbody tr")).toHaveCount(42);
    await page.getByRole("tab", { name: "Calibration" }).click();
    await expect(page.getByTestId("trajectories")).toContainText("Gen 4");
    await expect(
      page
        .getByTestId("overlays")
        .locator(".line-chart-wrapper, .bar-chart-wrapper"),
    ).toHaveCount(2);
    const posteriors = page.getByTestId("posteriors");
    await expect(posteriors).toContainText("Gen 4");
    await expect(posteriors.locator("tbody tr")).toHaveCount(2);
    await expect(posteriors.locator("tbody tr").first()).toContainText("r0");
    await expect(posteriors.locator("tbody tr").first()).toContainText(
      "Exponential(rate = 1)",
    );
    // Priors sit on their own tab.
    await expect(page.getByTestId("priors")).toBeHidden();
    await page.getByRole("tab", { name: "Priors" }).click();
    await expect(page.getByTestId("priors")).toContainText(
      "DiscreteUniform(a = 1, b = 4)",
    );
    await page.getByRole("tab", { name: "Calibration" }).click();
    // Projections sit on their own tab.
    await expect(page.getByTestId("projections")).toBeHidden();
    await page.getByRole("tab", { name: "Projections" }).click();
    await expect(page.getByTestId("projections")).toBeVisible();
    await expect(page.getByTestId("projections")).toContainText("baseline");
    await expect(page.getByTestId("projections")).toContainText("20 particles");
    await expect(page.getByTestId("trajectories")).toBeHidden();
    await page.getByRole("tab", { name: "Calibration" }).click();
    await expect(page.getByTestId("cells").locator(".gen-cell")).toHaveCount(5);

    await page.locator(".gen-table tbody tr").nth(1).click();
    await expect(page.getByTestId("trajectories")).toContainText("Gen 1");
  });

  test("the last run is reopened on the next visit until it is cleared", async ({
    page,
  }) => {
    await page.goto("/");
    await page.getByLabel("Load run.jsonl").setInputFiles(fixture);
    await expect(page.getByTestId("status")).toHaveText("Finished");

    await page.reload();
    await expect(page.getByTestId("status")).toHaveText("Finished");
    await expect(page.locator(".source-name")).toHaveText(
      "renewal.jsonl (saved copy)",
    );
    await expect(page.locator(".gen-table tbody tr")).toHaveCount(5);

    await page.getByRole("button", { name: "Clear" }).click();
    await expect(page.getByTestId("status")).toHaveText("No run loaded");
    await expect(page.getByRole("button", { name: "Clear" })).toHaveCount(0);

    await page.reload();
    await expect(page.getByLabel("Load run.jsonl")).toBeVisible();
    await expect(page.getByTestId("status")).toHaveText("No run loaded");
  });

  test("legend items turn trajectories off and on without moving the axes", async ({
    page,
  }) => {
    await page.goto("/");
    await page.getByLabel("Load run.jsonl").setInputFiles(fixture);
    await expect(page.getByTestId("status")).toHaveText("Finished");

    const section = page.getByTestId("trajectories");
    await expect(section).toContainText("20 rejected simulations");
    const lines = section.locator("svg path[stroke]");
    const all = await lines.count();
    const texts = () => section.locator("svg text").allTextContents();
    const before = await texts();
    const item = (label: string) =>
      section.locator("svg text").getByText(label, { exact: true });

    await item("Rejected").click();
    await expect(lines).toHaveCount(all - 20);
    await item("Accepted").click();
    await expect(lines).toHaveCount(all - 40);
    await item("Rejected").click();
    await expect(lines).toHaveCount(all - 20);

    // Hidden items stay in the legend, struck through, and the ticks hold.
    await expect(item("Accepted")).toHaveCSS(
      "text-decoration-line",
      "line-through",
    );
    await expect(item("Rejected")).toHaveCSS("text-decoration-line", "none");
    expect(await texts()).toEqual(before);
  });

  test("the generation selector shows one generation or all of them", async ({
    page,
  }) => {
    await page.goto("/");
    await page.getByLabel("Load run.jsonl").setInputFiles(fixture);
    await expect(page.getByTestId("status")).toHaveText("Finished");

    const section = page.getByTestId("trajectories");
    const subtitle = section.locator("h2 small");
    await expect(subtitle).toHaveText(
      "Gen 4, 20 of 100 accepted particles, 20 rejected simulations",
    );

    await section.getByRole("combobox", { name: "Generation" }).click();
    await page.getByRole("option", { name: "All generations" }).click();
    await expect(subtitle).toHaveText(
      "All generations, 100 of 500 accepted particles, 80 rejected simulations",
    );

    // Faceted, each generation gets its own chart on shared axes.
    const charts = section.locator(".line-chart-wrapper");
    await expect(charts).toHaveCount(1);
    await section.getByText("Facet by generation").click();
    await expect(charts).toHaveCount(5);
    await expect(charts.first()).toContainText(
      "Prior: 20 accepted, 0 rejected",
    );
    const ticks = (i: number) =>
      charts
        .nth(i)
        .locator("svg text")
        .filter({ hasText: /^[\d,.]+$/ })
        .allTextContents();
    expect(await ticks(0)).toEqual(await ticks(4));

    await section.getByRole("combobox", { name: "Generation" }).click();
    await page.getByRole("option", { name: "Generation 2" }).click();
    await expect(subtitle).toContainText("Gen 2, 20 of 100");
    await expect(charts).toHaveCount(1);
    await expect(section.getByText("Facet by generation")).toHaveCount(0);
    await expect(page.locator(".gen-table tbody tr.selected")).toContainText(
      "2",
    );

    // Picking a generation elsewhere leaves "All generations".
    await section.getByRole("combobox", { name: "Generation" }).click();
    await page.getByRole("option", { name: "All generations" }).click();
    await page.locator(".gen-table tbody tr").nth(1).click();
    await expect(subtitle).toContainText("Gen 1, 20 of 100");
  });

  test("chart tooltips name their series", async ({ page }) => {
    await page.goto("/");
    await page.getByLabel("Load run.jsonl").setInputFiles(fixture);
    await expect(page.getByTestId("status")).toHaveText("Finished");

    const hover = async (testId: string, fraction: number) => {
      const chart = page
        .getByTestId(testId)
        .locator(".line-chart-wrapper, .bar-chart-wrapper")
        .first();
      await chart.scrollIntoViewIfNeeded();
      const box = (await chart.boundingBox())!;
      await page.mouse.move(
        box.x + box.width * fraction,
        box.y + box.height * 0.5,
      );
      return chart.locator(".chart-tip");
    };

    const overlay = await hover("overlays", 0.45);
    await expect(overlay).toContainText("Density at r0 =");
    await expect(overlay).toContainText("Prior");
    await expect(overlay).toContainText("Gen 4");

    const cell = await hover("cells", 0.45);
    await expect(cell).toContainText("Mass at r0 ≈");

    // Inside the fitted window both series show; past it only the median.
    await page.getByRole("tab", { name: "Projections" }).click();
    const fitted = await hover("projections", 0.1);
    await expect(fitted).toContainText("Median");
    await expect(fitted).toContainText("Observed");
    const projected = await hover("projections", 0.6);
    await expect(projected).toContainText("Median");
    await expect(projected).not.toContainText("Observed");
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

    const posteriors = strFromU8(entries["posteriors.csv"]!).split("\n");
    expect(posteriors[0]).toBe(
      "generation,parameter,mean,sd,q05,q25,median,q75,q95",
    );
    expect(posteriors).toHaveLength(1 + 5 * 2);

    const plots = names.filter((n) => n.startsWith("plots/"));
    expect(charts).toBeGreaterThan(0);
    expect(plots).toHaveLength(charts);
    for (const name of plots) {
      expect(name).toMatch(/\.png$/);
      expect([...entries[name]!.slice(0, 4)]).toEqual([0x89, 0x50, 0x4e, 0x47]);
      expect(entries[name]!.length).toBeGreaterThan(1000);
    }
  });
});

test("an opened file is followed, and reopened on the next visit", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.getByLabel("Load run.jsonl")).toHaveCount(0);

  // A real handle, from the origin private file system, stands in for the
  // picker's.
  await page.evaluate(
    async (text) => {
      const root = await navigator.storage.getDirectory();
      const handle = await root.getFileHandle("run.jsonl", { create: true });
      const writable = await handle.createWritable();
      await writable.write(text);
      await writable.close();
      (
        window as unknown as { showOpenFilePicker: () => Promise<unknown[]> }
      ).showOpenFilePicker = async () => [handle];
    },
    await readFile(fixture, "utf8"),
  );

  await page.getByRole("button", { name: "Open run.jsonl" }).click();
  await expect(page.getByTestId("status")).toHaveText("Finished");
  await expect(page.locator(".source-name")).toHaveText("run.jsonl (watching)");

  await page.reload();
  await expect(page.getByTestId("status")).toHaveText("Finished");
  await expect(page.locator(".source-name")).toHaveText("run.jsonl (watching)");

  await page.getByRole("button", { name: "Clear" }).click();
  await expect(page.getByTestId("status")).toHaveText("No run loaded");
  await page.reload();
  await expect(
    page.getByRole("button", { name: "Open run.jsonl" }),
  ).toBeVisible();
  await expect(page.getByTestId("status")).toHaveText("No run loaded");
});
