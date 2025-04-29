<template>
    <div class="bg-yellow-200 text-yellow-900 p-3 rounded-md mb-4" v-if="show">
      ⚠️ Some filters were relaxed to find results.
    </div>
  </template>
  
  <script>
  export default {
    props: {
      explanation: {
        type: String,
        required: false
      }
    },
    computed: {
      show() {
        return this.explanation && this.explanation.toLowerCase().includes('relax')
      }
    }
  }
  </script>
  