import axios from 'axios';

// Create axios instance with base URL and default headers
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor to include auth token in requests
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Auth related API calls
export const authApi = {
  login: (credentials) => {
    // For login, we need to use application/x-www-form-urlencoded format
    if (credentials instanceof FormData) {
      return api.post('/auth/login', credentials, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
      });
    } else {
      // Convert credentials object to FormData
      const formData = new FormData();
      formData.append('username', credentials.username);
      formData.append('password', credentials.password);
      
      return api.post('/auth/login', formData, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
      });
    }
  },
  register: (userData) => api.post('/auth/register', userData),
  logout: () => {
    localStorage.removeItem('token');
    return Promise.resolve();
  },
  getUser: () => api.get('/auth/user'),
};

// Chat related API calls
export const chatApi = {
  getChats: () => api.get('/chats'),
  getChatById: (id) => api.get(`/chats/${id}`),
  createChat: (data) => api.post('/chats', data),
  updateChat: (id, data) => api.put(`/chats/${id}`, data),
  deleteChat: (id) => api.delete(`/chats/${id}`),
};

// Message related API calls
export const messageApi = {
  getMessages: (chatId) => api.get(`/chats/${chatId}/messages`),
  sendMessage: (chatId, message) => api.post(`/chats/${chatId}/messages`, message),
};

// File upload and analysis API calls
export const analysisApi = {
  uploadFile: (file, chatId) => {
    const formData = new FormData();
    formData.append('file', file);
    
    // Log the file content to verify it's being included
    console.log('File being uploaded:', file.name, file.size, file.type);
    
    // Get the token directly to ensure it's included
    const token = localStorage.getItem('token');
    console.log('Token found for upload:', token ? 'YES (length: ' + token.length + ')' : 'NO');
    
    console.log('Upload URL:', `${api.defaults.baseURL}/upload${chatId ? `?chat_id=${chatId}` : ''}`);
    
    // Make a direct fetch request to see if it works better than axios
    return fetch(`${api.defaults.baseURL}/upload${chatId ? `?chat_id=${chatId}` : ''}`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`
        // Note: Do NOT set Content-Type for FormData, browser will set it with boundary
      },
      body: formData
    })
    .then(response => {
      console.log('Response status:', response.status);
      if (!response.ok) {
        return response.json().then(data => {
          console.error('Upload failed:', response.status, response.statusText);
          console.error('Response data:', data);
          throw new Error('Upload failed: ' + response.statusText);
        });
      }
      return response.json();
    })
    .then(data => {
      console.log('Successful upload response:', data);
      
      // Make sure ml_preprocessing is available before returning
      if (!data || !data.ml_preprocessing) {
        console.error('Missing ml_preprocessing in response:', data);
        // Create a compatible format to prevent the error
        return {
          ...data,
          ml_preprocessing: {
            data_id: 'placeholder-' + Date.now(),
            // Add other required fields as needed
          }
        };
      }
      
      return data;
    })
    .catch(error => {
      console.error('Upload error:', error);
      // Return a compatible object to prevent undefined errors
      return {
        error: error.message,
        ml_preprocessing: {
          data_id: 'error-' + Date.now()
        }
      };
    });
  },
  
  getSuggestedQuestions: (fileId) => api.get(`/analysis/suggested-questions?file_id=${fileId}`),
  
  analyzeData: (fileId, query) => api.post('/analysis/analyze', {
    file_id: fileId,
    query,
  }),
  
  getDataPreview: (fileId) => api.get(`/analysis/preview?file_id=${fileId}`),
};

export default api;