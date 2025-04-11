import { useState, useEffect } from 'react';
import { AuthProvider, useAuth } from './context/AuthContext';
import { ChatProvider } from './context/ChatContext';
import Login from './components/auth/Login';
import Chat from './components/chat/Chat';
import './App.css';

// Main App component wrapper with providers
function App() {
  return (
    <AuthProvider>
      <ChatProvider>
        <AppContent />
      </ChatProvider>
    </AuthProvider>
  );
}

// App content that uses the context
function AppContent() {
  const { isAuthenticated, loading } = useAuth();
  const [appReady, setAppReady] = useState(false);

  // Add a slight delay to prevent flash of login screen
  useEffect(() => {
    const timer = setTimeout(() => {
      setAppReady(true);
    }, 500);

    return () => clearTimeout(timer);
  }, []);

  if (loading || !appReady) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-indigo-500"></div>
      </div>
    );
  }

  return isAuthenticated ? <Chat /> : <Login />;
}

export default App;
