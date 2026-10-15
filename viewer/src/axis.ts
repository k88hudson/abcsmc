// The x axis of the series charts. In model time a point sits at its index;
// in calendar time index `i` is drawn `i * stepDays` days after `start`. A
// run's log may give the calendar; the viewer's own choice is remembered per
// run where it differs from the log's.

import type { LoggedCalendar } from "./run";

export type TimeMode = "model" | "calendar";

export interface TimeAxis {
  mode: TimeMode;
  // yyyy-mm-dd, the date of index 0; "" until one is picked.
  start: string;
  stepDays: number;
}

export interface Calendar {
  startMs: number;
  stepDays: number;
}

const DAY_MS = 86_400_000;

export function defaultAxis(): TimeAxis {
  return { mode: "model", start: "", stepDays: 1 };
}

// The axis a run opens on: calendar time when its log gives a calendar.
export function logAxis(calendar: LoggedCalendar | null): TimeAxis {
  return calendar
    ? { mode: "calendar", start: calendar.start, stepDays: calendar.step_days }
    : defaultAxis();
}

export function sameAxis(a: TimeAxis, b: TimeAxis): boolean {
  return a.mode === b.mode && a.start === b.start && a.stepDays === b.stepDays;
}

// The calendar to draw on, or null for model time (also while calendar time
// has no valid start date yet).
export function calendarOf(axis: TimeAxis): Calendar | null {
  if (axis.mode !== "calendar") return null;
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(axis.start);
  if (!m) return null;
  const startMs = Date.UTC(+m[1]!, +m[2]! - 1, +m[3]!);
  if (new Date(startMs).toISOString().slice(0, 10) !== axis.start) return null;
  if (!(axis.stepDays > 0)) return null;
  return { startMs, stepDays: axis.stepDays };
}

export function msAt(c: Calendar, index: number): number {
  return c.startMs + index * c.stepDays * DAY_MS;
}

export function dateAt(c: Calendar, index: number): string {
  return new Date(msAt(c, index)).toISOString().slice(0, 10);
}

// A position on the chart's x axis: the index, or epoch ms on a date axis.
export function placeX(c: Calendar | null, index: number): number {
  return c ? msAt(c, index) : index;
}

// Series or areas with their `x` indices as dates, which puts the chart on a
// date axis. In model time they are returned as they are.
export function placeSeries<T extends { x?: number[] }>(
  c: Calendar | null,
  items: T[],
): (Omit<T, "x"> & { x?: number[] | string[] })[] {
  if (!c) return items;
  return items.map((item) =>
    item.x ? { ...item, x: item.x.map((i) => dateAt(c, i)) } : item,
  );
}

export function placeIntervals<T extends { x: number }>(
  c: Calendar | null,
  intervals: T[],
): T[] {
  if (!c) return intervals;
  return intervals.map((r) => ({ ...r, x: msAt(c, r.x) }));
}

const KEY = "abcsmc-viewer:time-axis:";

// Browser storage is a convenience: any failure reads as nothing saved.
export function loadAxis(runId: string): TimeAxis | null {
  try {
    const saved = JSON.parse(localStorage.getItem(KEY + runId) ?? "null");
    if (saved && (saved.mode === "model" || saved.mode === "calendar"))
      return {
        mode: saved.mode,
        start: typeof saved.start === "string" ? saved.start : "",
        stepDays: Number(saved.stepDays) > 0 ? Number(saved.stepDays) : 1,
      };
  } catch {
    // Nothing usable saved.
  }
  return null;
}

// Saves the viewer's choice, or forgets it when it is the log's own.
export function saveAxis(runId: string, axis: TimeAxis, own: TimeAxis): void {
  try {
    if (sameAxis(axis, own)) localStorage.removeItem(KEY + runId);
    else localStorage.setItem(KEY + runId, JSON.stringify(axis));
  } catch {
    // Not remembered next time.
  }
}
