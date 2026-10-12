<script setup lang="ts">
export interface PosteriorRow {
  parameter: string;
  // One per column: the shown generation's value, and the reference
  // generation's, drawn muted under it.
  cells: { value: string; reference: string }[];
}

defineProps<{
  columns: string[];
  rows: PosteriorRow[];
  reference: string;
}>();
</script>

<template>
  <div class="scroll">
    <table>
      <thead>
        <tr>
          <th>
            Parameter <span class="ref">({{ reference }} in grey)</span>
          </th>
          <th v-for="column in columns" :key="column" class="num">
            {{ column }}
          </th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in rows" :key="row.parameter">
          <td>{{ row.parameter }}</td>
          <td v-for="(cell, i) in row.cells" :key="i" class="num">
            {{ cell.value }}
            <span class="ref">{{ cell.reference }}</span>
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
  padding: 0.75em 1.25em;
  white-space: nowrap;
  text-align: left;
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
td,
th {
  vertical-align: top;
}
.ref {
  display: block;
  margin-top: 0.15em;
  font-size: 0.9em;
  font-weight: normal;
  color: var(--color-text-muted, #64748b);
}
th .ref {
  display: inline;
  margin: 0 0 0 0.3em;
}
</style>
