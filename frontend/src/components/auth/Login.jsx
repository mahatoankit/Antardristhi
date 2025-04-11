import { useState } from 'react';
import { authApi } from '../../api';
import { useAuth, USER_PROFILES } from '../../context/AuthContext';
import axios from 'axios';

const Login = () => {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isRegistering, setIsRegistering] = useState(false);
  const [fullName, setFullName] = useState('');
  const [selectedProfile, setSelectedProfile] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [loading, setLoading] = useState(false);
  const [debugInfo, setDebugInfo] = useState('');
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    setDebugInfo('');
    setLoading(true);
    
    try {
      if (isRegistering) {
        // Registration - direct approach to debug
        const userData = { 
          username, 
          email, 
          password, 
          full_name: fullName,
          profiles: selectedProfile ? [selectedProfile] : [] 
        };
        
        setDebugInfo(`Attempting to register with: ${JSON.stringify(userData)}`);
        console.log("Registering user with data:", userData);
        
        // Use direct axios call for debugging
        const baseURL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
        const response = await axios.post(`${baseURL}/auth/register`, userData, {
          headers: {
            'Content-Type': 'application/json',
          }
        });
        
        console.log("Registration response:", response);
        setDebugInfo(prev => `${prev}\nResponse: ${JSON.stringify(response.data)}`);
        setSuccess('Registration successful! You can now log in.');
        setIsRegistering(false);
      } else {
        // Login - using form data for OAuth2
        console.log("Logging in with:", { username, password });
        
        const formData = new URLSearchParams();
        formData.append('username', username);
        formData.append('password', password);
        
        const baseURL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
        const response = await axios.post(`${baseURL}/auth/login`, formData, {
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
          }
        });
        
        console.log("Login response:", response);
        localStorage.setItem('token', response.data.access_token);
        window.location.href = '/dashboard';
      }
    } catch (err) {
      console.error("Auth error:", err);
      setDebugInfo(prev => `${prev}\nError: ${err.message}`);
      
      if (err.response) {
        setDebugInfo(prev => `${prev}\nStatus: ${err.response.status}`);
        setDebugInfo(prev => `${prev}\nData: ${JSON.stringify(err.response.data)}`);
        setError(err.response.data?.detail || JSON.stringify(err.response.data) || 'Error in response from server');
      } else if (err.request) {
        setDebugInfo(prev => `${prev}\nNo response received from server`);
        setError('No response from server. Check if backend is running.');
      } else {
        setDebugInfo(prev => `${prev}\nRequest setup error: ${err.message}`);
        setError(`Error setting up request: ${err.message}`);
      }
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        <div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">
            {isRegistering ? 'Create your account' : 'Sign in to your account'}
          </h2>
        </div>
        
        {success && (
          <div className="rounded-md bg-green-50 p-4">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-green-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium text-green-800">{success}</p>
              </div>
            </div>
          </div>
        )}
        
        <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
          <input type="hidden" name="remember" defaultValue="true" />
          <div className="rounded-md shadow-sm -space-y-px">
            <div>
              <label htmlFor="username" className="sr-only">Username</label>
              <input
                id="username"
                name="username"
                type="text"
                required
                className="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-t-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 focus:z-10 sm:text-sm"
                placeholder="Username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
              />
            </div>
            
            {isRegistering && (
              <>
                <div>
                  <label htmlFor="email" className="sr-only">Email address</label>
                  <input
                    id="email"
                    name="email"
                    type="email"
                    required
                    className="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 focus:z-10 sm:text-sm"
                    placeholder="Email address"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                  />
                </div>
                <div>
                  <label htmlFor="fullName" className="sr-only">Full Name</label>
                  <input
                    id="fullName"
                    name="fullName"
                    type="text"
                    className="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 focus:z-10 sm:text-sm"
                    placeholder="Full Name"
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                  />
                </div>
                <div>
                  <label htmlFor="profile" className="sr-only">Profile</label>
                  <select
                    id="profile"
                    name="profile"
                    className="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 focus:z-10 sm:text-sm"
                    value={selectedProfile}
                    onChange={(e) => setSelectedProfile(e.target.value)}
                  >
                    <option value="">Select a profile</option>
                    {Object.entries(USER_PROFILES).map(([key, value]) => (
                      <option key={key} value={value}>
                        {value.replace('_', ' ')}
                      </option>
                    ))}
                  </select>
                </div>
              </>
            )}
            
            <div>
              <label htmlFor="password" className="sr-only">Password</label>
              <input
                id="password"
                name="password"
                type="password"
                required
                className="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-b-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 focus:z-10 sm:text-sm"
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>
          </div>
          
          {error && (
            <div className="rounded-md bg-red-50 p-4">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="ml-3">
                  <p className="text-sm font-medium text-red-800">{error}</p>
                </div>
              </div>
            </div>
          )}
          
          {debugInfo && (
            <div className="mt-4 p-3 bg-gray-100 rounded text-xs font-mono whitespace-pre-wrap overflow-auto max-h-40">
              {debugInfo}
            </div>
          )}
          
          <div>
            <button
              type="submit"
              disabled={loading}
              className="group relative w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
            >
              {loading ? 'Processing...' : (isRegistering ? 'Sign up' : 'Sign in')}
            </button>
          </div>
          
          <div className="flex items-center justify-center">
            <div className="text-sm">
              {isRegistering ? (
                <button
                  type="button"
                  className="font-medium text-indigo-600 hover:text-indigo-500"
                  onClick={() => setIsRegistering(false)}
                >
                  Already have an account? Sign in
                </button>
              ) : (
                <button
                  type="button"
                  className="font-medium text-indigo-600 hover:text-indigo-500"
                  onClick={() => setIsRegistering(true)}
                >
                  Don't have an account? Sign up
                </button>
              )}
            </div>
          </div>
        </form>
      </div>
    </div>
  );
};

export default Login;