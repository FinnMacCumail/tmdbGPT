<template>
    <div class="space-y-4">
      <ResultCard
        v-for="(entry, index) in entries"
        :key="index"
        :entry="entry"
      />
    </div>
  </template>
  
  <script>
  import ResultCard from './ResultCard.vue'
  
  export default {
    props: {
      entries: {
        type: Array,
        default: () => []
      }
    },
    components: {
      ResultCard
    }
  }
  </script>
  