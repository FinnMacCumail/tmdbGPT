import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api'; // adjust port as needed

export async function askQuery(userInput) {
  try {
    const response = await axios.post(`${API_BASE_URL}/ask`, {
      input: userInput,
    });
    return response.data;
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
}
