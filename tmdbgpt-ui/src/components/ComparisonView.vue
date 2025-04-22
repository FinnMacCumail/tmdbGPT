<template>
    <div class="comparison">
      <h2>🆚 {{ data.left.name }} vs {{ data.right.name }}</h2>
      <div class="columns">
        <div>
          <h3>{{ data.left.name }}</h3>
          <ul>
            <li v-for="(item, index) in data.left.entries" :key="index">
              <strong>{{ item.title }}</strong>
              <p>{{ item.overview }}</p>
            </li>
          </ul>
        </div>
        <div>
          <h3>{{ data.right.name }}</h3>
          <ul>
            <li v-for="(item, index) in data.right.entries" :key="index">
              <strong>{{ item.title }}</strong>
              <p>{{ item.overview }}</p>
            </li>
          </ul>
        </div>
      </div>
    </div>
  </template>
  
  <script>
  export default {
    name: 'ComparisonView',
    props: {
      data: Object
    }
  };
  </script>
  
  <style scoped>
  .comparison {
    margin-top: 2rem;
  }
  .columns {
    display: flex;
    gap: 2rem;
  }
  .columns > div {
    flex: 1;
  }
  </style>
  