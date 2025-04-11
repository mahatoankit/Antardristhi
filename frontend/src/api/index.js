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
    
    return api.post(`/analysis/upload?chat_id=${chatId}`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
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