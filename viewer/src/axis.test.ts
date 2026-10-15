import { describe, expect, it } from "vitest";
import {
  calendarOf,
  dateAt,
  defaultAxis,
  placeIntervals,
  placeSeries,
  placeX,
} from "./axis";

describe("calendarOf", () => {
  it("is null in model time or without a valid start date", () => {
    expect(calendarOf(defaultAxis())).toBeNull();
    expect(calendarOf({ mode: "calendar", start: "", stepDays: 1 })).toBeNull();
    expect(
      calendarOf({ mode: "calendar", start: "2026-02-30", stepDays: 1 }),
    ).toBeNull();
    expect(
      calendarOf({ mode: "calendar", start: "2026-02-01", stepDays: 0 }),
    ).toBeNull();
  });

  it("steps whole days from the start in UTC", () => {
    const daily = calendarOf({
      mode: "calendar",
      start: "2026-03-07",
      stepDays: 1,
    })!;
    expect(dateAt(daily, 0)).toBe("2026-03-07");
    // Across the US daylight saving change.
    expect(dateAt(daily, 2)).toBe("2026-03-09");
    expect(placeX(daily, 1)).toBe(Date.UTC(2026, 2, 8));
    const weekly = calendarOf({
      mode: "calendar",
      start: "2026-12-28",
      stepDays: 7,
    })!;
    expect(dateAt(weekly, 1)).toBe("2027-01-04");
  });
});

describe("placing", () => {
  const c = calendarOf({ mode: "calendar", start: "2026-01-01", stepDays: 1 });

  it("leaves everything as it is in model time", () => {
    const series = [{ x: [0, 1], data: [1, 2] }];
    expect(placeSeries(null, series)).toBe(series);
    expect(placeX(null, 3)).toBe(3);
  });

  it("turns series indices into dates and interval x into epoch ms", () => {
    expect(
      placeSeries(c, [{ x: [0, 2], data: [1, 2] }, { data: [3] }]),
    ).toEqual([
      { x: ["2026-01-01", "2026-01-03"], data: [1, 2] },
      { data: [3] },
    ]);
    expect(placeIntervals(c, [{ x: 1, lower: 0, upper: 1 }])).toEqual([
      { x: Date.UTC(2026, 0, 2), lower: 0, upper: 1 },
    ]);
  });
});
