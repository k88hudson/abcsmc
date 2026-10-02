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
    // The priors are on their own tab, not repeated here.
    await expect(posteriors).not.toContainText("Exponential");
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

  test("the posteriors table scrolls sideways on a narrow screen", async ({
    page,
  }) => {
    await page.setViewportSize({ width: 360, height: 800 });
    await page.goto("/");
    await page.getByLabel("Load run.jsonl").setInputFiles(fixture);
    await expect(page.getByTestId("status")).toHaveText("Finished");

    const wrapper = page.getByTestId("posteriors").locator(".TableWrapper");
    const sizes = await wrapper.evaluate((el) => ({
      scroll: el.scrollWidth,
      client: el.clientWidth,
      right: el.getBoundingClientRect().right,
    }));
    expect(sizes.scroll).toBeGreaterThan(sizes.client);
    expect(sizes.right).toBeLessThanOrEqual(360);

    await wrapper.evaluate((el) => (el.scrollLeft = el.scrollWidth));
    await expect(
      wrapper.getByRole("columnheader", { name: "95%" }),
    ).toBeInViewport({ ratio: 1 });
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

  test("legend items turn trajectories off and on, and the axes follow", async ({
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
    // The axes fit what is left, the observed series.
    expect(await texts()).not.toEqual(before);
    await item("Rejected").click();
    await expect(lines).toHaveCount(all - 20);

    // Hidden items stay in the legend, struck through.
    await expect(item("Accepted")).toHaveCSS(
      "text-decoration-line",
      "line-through",
    );
    await expect(item("Rejected")).toHaveCSS("text-decoration-line", "none");

    // With everything off the legend is still there to turn it back on.
    await item("Rejected").click();
    await item("Observed").click();
    await expect(item("Observed")).toHaveCSS(
      "text-decoration-line",
      "line-through",
    );
    for (const label of ["Observed", "Accepted", "Rejected"])
      await item(label).click();
    await expect(lines).toHaveCount(all);
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

  test("pairwise scatters report each pair's correlation", async ({ page }) => {
    await page.goto("/");
    await page.getByLabel("Load run.jsonl").setInputFiles(fixture);
    await expect(page.getByTestId("status")).toHaveText("Finished");

    // Two parameters make one pair; choosing the other one swaps the axes.
    const section = page.getByTestId("correlations");
    await expect(section.locator(".line-chart-wrapper")).toHaveCount(1);
    await expect(section).toContainText(
      /r0 ~ initial_infections: r = -?\d\.\d\d \(Gen 4\)/,
    );
    await expect(section.getByRole("combobox", { name: "Sort" })).toHaveCount(
      0,
    );
    await section.getByRole("combobox", { name: "Parameter" }).click();
    await page.getByRole("option", { name: "initial_infections" }).click();
    await expect(section).toContainText("initial_infections ~ r0: r =");
  });

  test("pairwise scatters show one parameter against the others, sorted", async ({
    page,
  }) => {
    // Three parameters: b follows a exactly, c runs against it loosely.
    const particles = Array.from({ length: 20 }, (_, i) => ({
      params: [i, 2 * i, 20 - i + (i % 3) * 4],
      weight: 1 / 20,
      distance: 0,
      seed: String(i),
    }));
    const stats = {
      tolerance: null,
      accepted: 20,
      attempts: 20,
      acceptance_ratio: 1,
      ess: 20,
      perplexity: 20,
      duration_seconds: 0,
    };
    const lines = [
      {
        type: "run_started",
        version: 1,
        id: "pairs",
        n_particles: 20,
        n_generations: 1,
        quantiles: null,
        params: ["a", "b", "c"].map((name) => ({ name, kind: "real" })),
        observed: null,
      },
      { type: "generation_started", generation: 0, tolerance: null },
      {
        type: "generation_completed",
        generation: 0,
        stats,
        particles,
        trajectories: [],
      },
      { type: "run_finished", generations: 1 },
    ];
    await page.goto("/");
    await page.getByLabel("Load run.jsonl").setInputFiles({
      name: "pairs.jsonl",
      mimeType: "application/jsonl",
      buffer: Buffer.from(lines.map((l) => JSON.stringify(l)).join("\n")),
    });
    await expect(page.getByTestId("status")).toHaveText("Finished");

    const section = page.getByTestId("correlations");
    const charts = section.locator(".line-chart-wrapper");
    // The first parameter against the other two, strongest first.
    await expect(charts).toHaveCount(2);
    await expect(charts.first()).toContainText("a ~ b: r = 1.00");

    await section.getByRole("combobox", { name: "Sort" }).click();
    await page.getByRole("option", { name: "Weakest |r| first" }).click();
    await expect(charts.last()).toContainText("a ~ b: r = 1.00");

    await section.getByRole("combobox", { name: "Parameter" }).click();
    await page.getByRole("option", { name: "c", exact: true }).click();
    await expect(charts).toHaveCount(2);
    await expect(section).toContainText("c ~ a");
    await expect(section).toContainText("c ~ b");
    await expect(section).not.toContainText("a ~ b");
  });

  test("dragging across a chart zooms it, and the reset button undoes it", async ({
    page,
  }) => {
    await page.goto("/");
    await page.getByLabel("Load run.jsonl").setInputFiles(fixture);
    await expect(page.getByTestId("status")).toHaveText("Finished");

    const chart = page
      .getByTestId("trajectories")
      .locator(".line-chart-wrapper");
    await chart.scrollIntoViewIfNeeded();
    const ticks = () =>
      chart
        .locator("svg text")
        .filter({ hasText: /^[\d,.]+$/ })
        .allTextContents();
    const before = await ticks();
    const reset = chart.getByRole("button", { name: /reset/i });
    await expect(reset).toHaveCount(0);

    const box = (await chart.boundingBox())!;
    const y = box.y + box.height * 0.6;
    await page.mouse.move(box.x + box.width * 0.4, y);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width * 0.6, y, { steps: 5 });
    await page.mouse.up();
    await expect(reset).toBeVisible();
    expect(await ticks()).not.toEqual(before);

    await reset.click();
    await expect(reset).toHaveCount(0);
    expect(await ticks()).toEqual(before);
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
