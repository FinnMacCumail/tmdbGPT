<template>
    <div class="p-6 max-w-4xl mx-auto space-y-6">
      <h1 class="text-2xl font-bold text-center">🎬 TMDB GPT</h1>
  
      <!-- Query input -->
      <QueryInput @submit="askQuery" :loading="loading" />
  
      <!-- Fallback warning -->
      <FallbackBanner v-if="response?.explanation?.includes('Relaxed')" :explanation="response.explanation" />
  
      <!-- Error alert -->
      <div v-if="error" class="bg-red-100 text-red-700 p-4 rounded">
        ⚠️ Error: {{ error }}
      </div>
  
      <!-- Loading state -->
      <LoadingSpinner v-if="loading" />
  
      <!-- Results section -->
      <div v-if="!loading && entries.length" class="space-y-4">
        <!-- Dynamic layout rendering -->
        <div v-if="responseFormat === 'comparison'" class="grid grid-cols-2 gap-4">
          <div>
            <h2 class="font-bold">{{ response.left.name }}</h2>
            <ResultCard v-for="entry in response.left.entries" :key="entry.title" :entry="entry" />
          </div>
          <div>
            <h2 class="font-bold">{{ response.right.name }}</h2>
            <ResultCard v-for="entry in response.right.entries" :key="entry.title" :entry="entry" />
          </div>
        </div>
  
        <div v-else-if="responseFormat === 'timeline'">
          <div v-for="entry in response.entries" :key="entry.title">
            <div class="text-gray-500 text-sm">{{ entry.release_year }}</div>
            <ResultCard :entry="entry" />
          </div>
        </div>
  
        <div v-else>
          <ResultCard v-for="entry in entries" :key="entry" :entry="{ title: entry }" />
          <DebugDetails v-if="response?.execution_trace" :trace="response.execution_trace" />
        </div>
      </div>
  
      <!-- Debug info -->
      <details v-if="response?.execution_trace" class="mt-6 text-sm bg-gray-100 p-2 rounded">
        <summary class="cursor-pointer font-semibold">Debug Details</summary>
        <pre class="overflow-auto max-h-64">{{ JSON.stringify(response.execution_trace, null, 2) }}</pre>
      </details>
    </div>
  </template>
  
  <script>
  import QueryInput from '@/components/QueryInput.vue'
  import ResultCard from '@/components/ResultCard.vue'
  import FallbackBanner from '@/components/FallbackBanner.vue'
  import LoadingSpinner from '@/components/LoadingSpinner.vue'
  import { askQuery } from '@/services/api.js'
  
  export default {
    components: {
      QueryInput,
      ResultCard,
      FallbackBanner,
      LoadingSpinner
    },
    data() {
      return {
        loading: false,
        response: null,
        error: null
      }
    },
    computed: {
      entries() {
        return this.response?.entries || []
      },
      responseFormat() {
        return this.response?.response_format || 'ranked_list'
      }
    },
    methods: {
      async askQuery(query) {
        this.loading = true
        this.error = null
        this.response = null
  
        try {
          const res = await askQuery(query)
          if (res.status === 'ok') {
            this.response = res.response
          } else {
            this.error = res.error || 'Unknown error'
          }
        } catch (e) {
          this.error = e.message || 'Request failed.'
        } finally {
          this.loading = false
        }
      }
    }
  }
  </script>
  
  <style scoped>
  /* Add any scoped styles here */
  </style>
  