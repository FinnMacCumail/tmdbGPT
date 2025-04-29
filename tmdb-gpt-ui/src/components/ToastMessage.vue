<template>
    <div v-if="visible" class="fixed bottom-4 right-4 bg-red-500 text-white px-4 py-2 rounded shadow-lg animate-fade">
      {{ message }}
    </div>
  </template>
  
  <script>
  export default {
    props: {
      message: {
        type: String,
        required: true
      }
    },
    data() {
      return {
        visible: false
      }
    },
    watch: {
      message(newVal) {
        if (newVal) {
          this.visible = true
          setTimeout(() => {
            this.visible = false
          }, 3000)
        }
      }
    }
  }
  </script>
  
  <style scoped>
  @keyframes fade {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
  }
  
  .animate-fade {
    animation: fade 0.5s ease-out;
  }
  </style>
  