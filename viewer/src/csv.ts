type Cell = number | string | null | undefined;

function escapeField(value: string): string {
  return /[",\n]/.test(value) ? `"${value.replace(/"/g, '""')}"` : value;
}

// Column-oriented CSV: one header per column, rows run to the longest column
// and shorter columns leave their cells empty.
export function columnsToCsv(
  columns: { header: string; values: readonly Cell[] }[],
): string {
  let length = 0;
  for (const c of columns) length = Math.max(length, c.values.length);
  const rows = [columns.map((c) => escapeField(c.header)).join(",")];
  for (let r = 0; r < length; r++) {
    rows.push(
      columns
        .map((c) => {
          const v = c.values[r];
          return v === null || v === undefined ? "" : escapeField(String(v));
        })
        .join(","),
    );
  }
  return rows.join("\n");
}

// A filename stem from free text such as a run id, label, or parameter name.
export function fileStem(...parts: Cell[]): string {
  return parts
    .filter((p) => p !== null && p !== undefined && p !== "")
    .map((p) =>
      String(p)
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-+|-+$/g, ""),
    )
    .filter((p) => p !== "")
    .join("-");
}
