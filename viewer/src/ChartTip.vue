<script setup lang="ts">
// Tooltip body for a chart's #tooltip slot: a heading and one named row per
// series, since the default tooltip shows values without their series names.
defineProps<{
  heading: string;
  values: { value: number; color: string; seriesIndex: number }[];
  labels: (string | undefined)[];
  digits?: number;
}>();

function format(x: number, digits: number): string {
  if (!Number.isFinite(x)) return "–";
  return Number.isInteger(x) ? String(x) : x.toFixed(digits);
}
</script>

<template>
  <div class="chart-tip">
    <div class="chart-tip__heading">{{ heading }}</div>
    <div v-for="v in values" :key="v.seriesIndex" class="chart-tip__row">
      <span class="chart-tip__swatch" :style="{ background: v.color }" />
      <span class="chart-tip__label">{{
        labels[v.seriesIndex] ?? `Series ${v.seriesIndex + 1}`
      }}</span>
      <span class="chart-tip__value">{{ format(v.value, digits ?? 3) }}</span>
    </div>
  </div>
</template>

<style scoped>
.chart-tip {
  white-space: nowrap;
}
.chart-tip__heading {
  font-weight: 600;
  margin-bottom: 0.25em;
}
.chart-tip__row {
  display: flex;
  align-items: center;
  gap: 0.4em;
}
.chart-tip__swatch {
  width: 0.625em;
  height: 0.625em;
  border-radius: 50%;
  flex-shrink: 0;
}
.chart-tip__value {
  margin-left: auto;
  padding-left: 1em;
  font-variant-numeric: tabular-nums;
}
</style>
