<template>
    <div class="ranked-list">
      <h2>🏆 Top Results</h2>
      <ol>
        <li v-for="(entry, index) in data.entries" :key="index">{{ entry }}</li>
      </ol>
    </div>
  </template>
  
  <script>
  export default {
    name: 'RankedListView',
    props: {
      data: Object
    }
  };
  </script>
  
  <style scoped>
  .ranked-list {
    margin-top: 1.5rem;
  }
  .ranked-list ol {
    padding-left: 1.5rem;
  }
  </style>
  