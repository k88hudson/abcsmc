import { describe, expect, it } from "vitest";
import { columnsToCsv, fileStem } from "./csv";

describe("columnsToCsv", () => {
  it("pads short columns and escapes fields", () => {
    const csv = columnsToCsv([
      { header: "index", values: [0, 1, 2] },
      { header: "observed", values: [5, null] },
      { header: 'r0, "scaled"', values: ["a,b", "c", "d"] },
    ]);
    expect(csv).toBe(
      ['index,observed,"r0, ""scaled"""', '0,5,"a,b"', "1,,c", "2,,d"].join(
        "\n",
      ),
    );
  });
});

describe("fileStem", () => {
  it("joins slugged parts and skips empty ones", () => {
    expect(fileStem("renewal", null, "Transmission -30% after", "r0")).toBe(
      "renewal-transmission-30-after-r0",
    );
  });
});
