<template>
    <div class="timeline">
      <h2>🕒 {{ data.entity }} Timeline</h2>
      <ul>
        <li v-for="(entry, index) in data.entries" :key="index">
          <strong>{{ entry.release_year }} - {{ entry.title }}</strong>
          <p>{{ entry.overview }}</p>
        </li>
      </ul>
    </div>
  </template>
  
  <script>
  export default {
    name: 'TimelineView',
    props: {
      data: Object
    }
  };
  </script>
  
  <style scoped>
  .timeline {
    margin-top: 1.5rem;
  }
  .timeline ul {
    list-style: none;
    padding-left: 0;
  }
  .timeline li {
    margin-bottom: 1.2rem;
  }
  </style>
  