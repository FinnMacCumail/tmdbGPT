import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000'; // adjust port as needed

export async function askQuery(userInput) {
  try {
    const response = await axios.post(`${API_BASE_URL}/query`, {
      query: userInput, // ✅ matches your FastAPI model: QueryRequest(query: str)    
    });
    return response.data;
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
}
