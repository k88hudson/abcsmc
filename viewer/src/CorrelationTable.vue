<script setup lang="ts">
export interface CorrelationRow {
  key: string;
  label: string;
  r: number;
  referenceR: number;
}

defineProps<{
  rows: CorrelationRow[];
  reference: string | null;
  color: string;
}>();
const emit = defineEmits<{ select: [key: string] }>();

// |r| at or above this is drawn in bold.
const STRONG = 0.5;

function fmt(r: number): string {
  return Number.isFinite(r) ? r.toFixed(2) : "–";
}

// A bar out from the middle of its track: right for a positive r, left for a
// negative one.
function bar(r: number): Record<string, string> {
  const half = Number.isFinite(r) ? Math.abs(r) * 50 : 0;
  return { left: `${r < 0 ? 50 - half : 50}%`, width: `${half}%` };
}
</script>

<template>
  <div class="scroll">
    <table>
      <thead>
        <tr>
          <th>Pair</th>
          <th class="num">
            r
            <span v-if="reference" class="ref">({{ reference }} in grey)</span>
          </th>
          <th><span class="visually-hidden">Strength</span></th>
        </tr>
      </thead>
      <tbody>
        <tr
          v-for="row in rows"
          :key="row.key"
          :class="{ strong: Math.abs(row.r) >= STRONG }"
        >
          <td>
            <button
              type="button"
              class="pair"
              title="Show this pair's plot"
              @click="emit('select', row.key)"
            >
              {{ row.label }}
            </button>
          </td>
          <td class="num">
            <span class="r">{{ fmt(row.r) }}</span>
            <span v-if="reference" class="ref block">{{
              fmt(row.referenceR)
            }}</span>
          </td>
          <td>
            <div class="track" aria-hidden="true">
              <div class="fill" :style="{ ...bar(row.r), background: color }" />
              <div
                v-if="Number.isFinite(row.referenceR)"
                class="tick"
                :style="{ left: `${50 + row.referenceR * 50}%` }"
              />
            </div>
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<style scoped>
.scroll {
  max-width: 100%;
  overflow-x: auto;
  font-size: var(--font-size-sm);
}
table {
  margin: 0;
  border-collapse: collapse;
  font-variant-numeric: tabular-nums;
  border: 1px solid var(--color-border);
}
th,
td {
  padding: 0.6em 1.25em;
  white-space: nowrap;
  text-align: left;
  vertical-align: middle;
  background: transparent;
}
th {
  font-weight: 600;
  border-bottom: 1px solid var(--color-border-header);
}
tbody td {
  border-bottom: 1px solid var(--color-border);
}
tbody tr:last-child td {
  border-bottom: none;
}
.num {
  text-align: right;
}
.strong .r {
  font-weight: 700;
}
.ref {
  margin-left: 0.3em;
  font-size: 0.9em;
  font-weight: normal;
  color: var(--color-text-muted, #64748b);
}
.ref.block {
  display: block;
  margin: 0.15em 0 0;
}
.pair {
  padding: 0;
  border: none;
  background: none;
  color: inherit;
  font: inherit;
  text-align: left;
  cursor: pointer;
}
.pair:hover {
  text-decoration: underline;
}
.track {
  position: relative;
  width: 8rem;
  height: 0.6rem;
  border-radius: 0.3rem;
  background: var(--color-bg-1, rgba(0, 0, 0, 0.06));
}
.track::after {
  content: "";
  position: absolute;
  left: 50%;
  top: -0.15rem;
  bottom: -0.15rem;
  border-left: 1px solid var(--color-border-header, #94a3b8);
}
.fill {
  position: absolute;
  top: 0;
  bottom: 0;
  border-radius: 0.3rem;
}
.tick {
  position: absolute;
  top: -0.2rem;
  bottom: -0.2rem;
  width: 2px;
  margin-left: -1px;
  background: var(--color-text-muted, #64748b);
}
.visually-hidden {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip-path: inset(50%);
  white-space: nowrap;
}
</style>
