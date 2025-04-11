import { createContext, useContext, useState, useEffect } from 'react';
import { useAuth } from './AuthContext';

// Create the chat context
const ChatContext = createContext();

export const ChatProvider = ({ children }) => {
  const { user } = useAuth();
  const [chats, setChats] = useState([]);
  const [activeChat, setActiveChat] = useState(null);
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [dataFile, setDataFile] = useState(null);
  const [suggestedQuestions, setSuggestedQuestions] = useState([]);
  const [analysisResults, setAnalysisResults] = useState(null);
  
  // Load chat history from localStorage when component mounts or user changes
  useEffect(() => {
    if (user) {
      const storedChats = localStorage.getItem(`chats_${user.id}`);
      if (storedChats) {
        const parsedChats = JSON.parse(storedChats);
        setChats(parsedChats);
        
        // Set the active chat to the most recent one or create a new one
        if (parsedChats.length > 0) {
          const lastChat = parsedChats[parsedChats.length - 1];
          setActiveChat(lastChat);
          setMessages(lastChat.messages || []);
        } else {
          createNewChat();
        }
      } else {
        createNewChat();
      }
    }
  }, [user]);
  
  // Save chats to localStorage whenever they change
  useEffect(() => {
    if (user && chats.length > 0) {
      localStorage.setItem(`chats_${user.id}`, JSON.stringify(chats));
    }
  }, [chats, user]);
  
  // Create a new chat
  const createNewChat = () => {
    const newChat = {
      id: Date.now().toString(),
      title: `Chat ${chats.length + 1}`,
      createdAt: new Date().toISOString(),
      messages: [],
      dataFile: null,
    };
    
    setChats(prevChats => [...prevChats, newChat]);
    setActiveChat(newChat);
    setMessages([]);
    setDataFile(null);
    setSuggestedQuestions([]);
    setAnalysisResults(null);
    
    return newChat;
  };
  
  // Switch to a different chat
  const switchChat = (chatId) => {
    const chat = chats.find(c => c.id === chatId);
    if (chat) {
      setActiveChat(chat);
      setMessages(chat.messages || []);
      setDataFile(chat.dataFile || null);
      setSuggestedQuestions(chat.suggestedQuestions || []);
      setAnalysisResults(chat.analysisResults || null);
    }
  };
  
  // Delete a chat
  const deleteChat = (chatId) => {
    const updatedChats = chats.filter(chat => chat.id !== chatId);
    setChats(updatedChats);
    
    // If the active chat was deleted, switch to the most recent one or create a new one
    if (activeChat && activeChat.id === chatId) {
      if (updatedChats.length > 0) {
        const lastChat = updatedChats[updatedChats.length - 1];
        setActiveChat(lastChat);
        setMessages(lastChat.messages || []);
        setDataFile(lastChat.dataFile || null);
        setSuggestedQuestions(lastChat.suggestedQuestions || []);
        setAnalysisResults(lastChat.analysisResults || null);
      } else {
        createNewChat();
      }
    }
  };
  
  // Rename a chat
  const renameChat = (chatId, newTitle) => {
    const updatedChats = chats.map(chat => {
      if (chat.id === chatId) {
        return { ...chat, title: newTitle };
      }
      return chat;
    });
    
    setChats(updatedChats);
    
    if (activeChat && activeChat.id === chatId) {
      setActiveChat({ ...activeChat, title: newTitle });
    }
  };
  
  // Add a message to the current chat
  const addMessage = (message) => {
    const updatedMessages = [...messages, message];
    setMessages(updatedMessages);
    
    // Update the messages in the active chat
    if (activeChat) {
      const updatedChat = { ...activeChat, messages: updatedMessages };
      const updatedChats = chats.map(chat => {
        if (chat.id === activeChat.id) {
          return updatedChat;
        }
        return chat;
      });
      
      setChats(updatedChats);
      setActiveChat(updatedChat);
    }
    
    return updatedMessages;
  };
  
  // Set the data file for the current chat
  const setCurrentDataFile = (file) => {
    setDataFile(file);
    
    // Update the data file in the active chat
    if (activeChat) {
      const updatedChat = { ...activeChat, dataFile: file };
      const updatedChats = chats.map(chat => {
        if (chat.id === activeChat.id) {
          return updatedChat;
        }
        return chat;
      });
      
      setChats(updatedChats);
      setActiveChat(updatedChat);
    }
  };
  
  // Set suggested questions for the current chat
  const setCurrentSuggestedQuestions = (questions) => {
    setSuggestedQuestions(questions);
    
    // Update the suggested questions in the active chat
    if (activeChat) {
      const updatedChat = { ...activeChat, suggestedQuestions: questions };
      const updatedChats = chats.map(chat => {
        if (chat.id === activeChat.id) {
          return updatedChat;
        }
        return chat;
      });
      
      setChats(updatedChats);
      setActiveChat(updatedChat);
    }
  };
  
  // Set analysis results for the current chat
  const setCurrentAnalysisResults = (results) => {
    setAnalysisResults(results);
    
    // Update the analysis results in the active chat
    if (activeChat) {
      const updatedChat = { ...activeChat, analysisResults: results };
      const updatedChats = chats.map(chat => {
        if (chat.id === activeChat.id) {
          return updatedChat;
        }
        return chat;
      });
      
      setChats(updatedChats);
      setActiveChat(updatedChat);
    }
  };
  
  // Context value
  const value = {
    chats,
    activeChat,
    messages,
    loading,
    dataFile,
    suggestedQuestions,
    analysisResults,
    setLoading,
    createNewChat,
    switchChat,
    deleteChat,
    renameChat,
    addMessage,
    setCurrentDataFile,
    setCurrentSuggestedQuestions,
    setCurrentAnalysisResults,
  };
  
  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
};

// Custom hook to use the chat context
export const useChat = () => {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return context;
};