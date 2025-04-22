<template>
    <div class="summary">
      <p>{{ data.text }}</p>
    </div>
  </template>
  
  <script>
  export default {
    name: 'SummaryView',
    props: {
      data: Object
    }
  };
  </script>
  
  <style scoped>
  .summary {
    margin-top: 1.5rem;
    font-size: 1.2rem;
    line-height: 1.6;
  }
  </style>
  