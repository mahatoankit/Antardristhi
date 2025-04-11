import { createContext, useContext, useState, useEffect } from 'react';
import { authApi } from '../api';

// Create the authentication context
const AuthContext = createContext();

// User profiles
export const USER_PROFILES = {
  STUDENT: 'student',
  BUSINESS_OWNER: 'business_owner',
  DEVELOPER: 'developer',
  ANALYST: 'analyst',
  GENERAL: 'general'
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Load user data from local storage when the component mounts
  useEffect(() => {
    const loadUser = async () => {
      const token = localStorage.getItem('token');
      if (token) {
        try {
          const response = await authApi.getUser();
          setUser(response.data);
        } catch (err) {
          console.error('Failed to load user:', err);
          localStorage.removeItem('token');
        }
      }
      setLoading(false);
    };
    
    loadUser();
  }, []);
  
  // Login function
  const login = async (credentials) => {
    try {
      setError(null);
      const response = await authApi.login(credentials);
      const { access_token, user: userData } = response.data;
      
      // Store token
      localStorage.setItem('token', access_token);
      
      // Store user data
      setUser(userData);
      
      return userData;
    } catch (err) {
      console.error('Login error:', err);
      setError(err.response?.data?.detail || 'Login failed. Please try again.');
      throw err;
    }
  };
  
  // Register function
  const register = async (userData) => {
    try {
      setError(null);
      const response = await authApi.register(userData);
      return response.data;
    } catch (err) {
      console.error('Registration error:', err);
      setError(err.response?.data?.detail || 'Registration failed. Please try again.');
      throw err;
    }
  };
  
  // Logout function
  const logout = () => {
    setUser(null);
    localStorage.removeItem('token');
    localStorage.removeItem('user');
  };
  
  // Change user profile
  const changeProfile = (profile) => {
    if (user) {
      const updatedUser = { ...user, profile };
      setUser(updatedUser);
      return updatedUser;
    }
    return null;
  };
  
  // Context value
  const value = {
    user,
    loading,
    error,
    login,
    register,
    logout,
    changeProfile,
    isAuthenticated: !!user,
  };
  
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

// Custom hook to use the auth context
export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};