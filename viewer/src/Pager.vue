<script setup lang="ts">
import { Button } from "cfasim-ui/components";

const page = defineModel<number>({ required: true });
defineProps<{ pages: number; total: number; noun: string }>();
</script>

<template>
  <nav v-if="pages > 1" class="pager" aria-label="Pages">
    <Button
      variant="secondary"
      :disabled="page === 0"
      aria-label="Previous page"
      @click="page--"
      >‹ Prev</Button
    >
    <span data-testid="page-status"
      >Page {{ page + 1 }} of {{ pages }}
      <span class="muted">({{ total }} {{ noun }})</span></span
    >
    <Button
      variant="secondary"
      :disabled="page >= pages - 1"
      aria-label="Next page"
      @click="page++"
      >Next ›</Button
    >
  </nav>
</template>

<style scoped>
.pager {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.75rem;
  margin: 0.75rem 0;
  font-size: 0.85rem;
}
.muted {
  color: var(--color-text-muted, #64748b);
}
</style>
