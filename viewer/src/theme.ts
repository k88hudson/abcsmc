import { onScopeDispose, ref, type Ref } from "vue";

export interface ChartPalette {
  prior: string;
  posterior: string;
  // Overlapping bars darken on a light surface and lighten on a dark one.
  overlapBlend: "multiply" | "screen";
  kde: string;
  observed: string;
  trajectory: string;
  median: string;
  band: string;
  bandOpacity: [number, number];
}

const light: ChartPalette = {
  prior: "#94a3b8",
  posterior: "#2563eb",
  overlapBlend: "multiply",
  kde: "#dc2626",
  observed: "#14b8a6",
  trajectory: "rgba(100, 116, 139, 0.3)",
  median: "#1d4ed8",
  band: "#2563eb",
  bandOpacity: [0.15, 0.3],
};

const dark: ChartPalette = {
  prior: "#5f6875",
  posterior: "#3987e5",
  overlapBlend: "screen",
  kde: "#e66767",
  observed: "#2dd4bf",
  trajectory: "rgba(148, 163, 184, 0.3)",
  median: "#86b6ef",
  band: "#3987e5",
  bandOpacity: [0.2, 0.4],
};

export function chartPalette(isDark: boolean): ChartPalette {
  return isDark ? dark : light;
}

// Generation 0 is grey (the prior); later generations run in blue toward
// higher contrast with the surface: darker in light mode, lighter in dark.
export function generationColor(i: number, n: number, isDark = false): string {
  if (i === 0) return isDark ? "#6b7280" : "#9ca3af";
  const t = n > 1 ? i / (n - 1) : 0;
  const lightness = isDark ? 38 + t * 40 : 70 - t * 52;
  return `hsl(${isDark ? 214 : 222}, 75%, ${Math.round(lightness)}%)`;
}

// Follows the theme's convention: a `dark` or `light` class on the root wins,
// otherwise the OS preference applies.
export function useDark(): Ref<boolean> {
  const root = document.documentElement;
  const media = window.matchMedia("(prefers-color-scheme: dark)");
  const read = () =>
    root.classList.contains("dark") ||
    (!root.classList.contains("light") && media.matches);
  const isDark = ref(read());
  const update = () => {
    isDark.value = read();
  };
  media.addEventListener("change", update);
  const observer = new MutationObserver(update);
  observer.observe(root, { attributes: true, attributeFilter: ["class"] });
  onScopeDispose(() => {
    media.removeEventListener("change", update);
    observer.disconnect();
  });
  return isDark;
}
