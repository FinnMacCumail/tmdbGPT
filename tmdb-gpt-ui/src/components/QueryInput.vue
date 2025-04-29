<template>
    <form @submit.prevent="handleSubmit" class="flex gap-2">
      <input
        v-model="input"
        type="text"
        placeholder="Ask a movie or TV question..."
        class="flex-1 border rounded p-2"
        :disabled="loading"
      />
      <button
        type="submit"
        class="bg-blue-600 text-white px-4 py-2 rounded"
        :disabled="loading || !input.trim()"
      >
        Search
      </button>
    </form>
  </template>
  
  <script>
  export default {
    props: {
      loading: {
        type: Boolean,
        default: false
      }
    },
    data() {
      return {
        input: ''
      }
    },
    methods: {
      handleSubmit() {
        this.$emit('submit', this.input.trim())
      }
    }
  }
  </script>
  
  <style scoped>
  input:disabled {
    background-color: #f3f3f3;
    cursor: not-allowed;
  }
  </style>