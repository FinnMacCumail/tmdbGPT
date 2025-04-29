<template>
    <div class="border rounded p-4 shadow-sm bg-white">
      <h3 class="text-lg font-semibold mb-1">
        {{ badge }} {{ entry.title }}
      </h3>
      <p class="text-sm text-gray-700 mb-1">{{ entry.overview }}</p>
      <p v-if="isRelaxed" class="text-xs text-yellow-700">⚠️ Result from relaxed fallback</p>
    </div>
  </template>
  
  <script>
  export default {
    props: {
      entry: {
        type: Object,
        required: true
      }
    },
    computed: {
      badge() {
        const source = this.entry.source || ''
        if (source.includes('/tv')) return '📺'
        if (source.includes('/movie')) return '🎬'
        return '🎞️'
      },
      isRelaxed() {
        return this.entry.source && this.entry.source.includes('_relaxed')
      }
    }
  }
  </script>
  