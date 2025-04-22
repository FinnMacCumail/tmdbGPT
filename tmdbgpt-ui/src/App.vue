<template>
  <div class="app">
    <h1>🎬 TMDBGPT</h1>
    <input
      v-model="query"
      @keyup.enter="runQuery"
      placeholder="Ask a TMDB question..."
    />
    <button @click="runQuery">Submit</button>

    <div v-if="loading" class="status">⏳ Loading...</div>
    <div v-if="error" class="status error">⚠️ {{ error }}</div>

    <div v-if="response">
      <component :is="getComponent(response.response_format)" :data="response" />
      <details class="debug" v-if="debug">
        <summary>🧠 Debug Info</summary>
        <pre>{{ debug }}</pre>
      </details>
    </div>
  </div>
</template>

<script>
import { fetchTMDBResponse } from './api.js'
import SummaryView from './components/SummaryView.vue'
import TimelineView from './components/TimelineView.vue'
import RankedListView from './components/RankedListView.vue'
import ComparisonView from './components/ComparisonView.vue'

export default {
  components: {
    SummaryView,
    TimelineView,
    RankedListView,
    ComparisonView
  },
  data() {
    return {
      query: '',
      response: null,
      error: null,
      loading: false,
      debug: null
    };
  },
  methods: {
    async runQuery() {
      this.loading = true;
      this.response = null;
      this.error = null;
      this.debug = null;
      try {
        const result = await fetchTMDBResponse(this.query);
        if (result.status === 'error') {
          this.error = result.error;
        } else {
          this.response = result.response;
          this.debug = result.trace || null;
        }
      } catch (err) {
        this.error = err.message || 'Unexpected error occurred.';
      } finally {
        this.loading = false;
      }
    },
    getComponent(type) {
      return {
        summary: 'SummaryView',
        count_summary: 'SummaryView',
        timeline: 'TimelineView',
        ranked_list: 'RankedListView',
        comparison: 'ComparisonView'
      }[type] || 'SummaryView';
    }
  }
};
</script>

<style>
.app {
  padding: 2rem;
  font-family: 'Segoe UI', sans-serif;
  background: #f5f5f5;
  color: #333;
  max-width: 800px;
  margin: 0 auto;
}

input {
  width: 70%;
  padding: 0.6rem;
  margin-right: 1rem;
  border: 1px solid #ccc;
  border-radius: 4px;
}

button {
  padding: 0.6rem 1.2rem;
  background: #333;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.status {
  margin-top: 1rem;
  font-style: italic;
}

.status.error {
  color: red;
}

.debug {
  margin-top: 2rem;
  background: #eee;
  padding: 1rem;
  border-radius: 6px;
  font-size: 0.9rem;
}
</style>
