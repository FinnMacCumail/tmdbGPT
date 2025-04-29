<template>
    <details class="bg-gray-100 p-3 rounded mt-6">
      <summary class="cursor-pointer font-semibold">Execution Trace</summary>
      <pre class="overflow-auto text-sm mt-2">
  {{ formattedTrace }}
      </pre>
    </details>
  </template>
  
  <script>
  export default {
    props: {
      trace: {
        type: Array,
        required: true
      }
    },
    computed: {
      formattedTrace() {
        return JSON.stringify(this.trace, null, 2)
      }
    }
  }
  </script>