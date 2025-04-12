import { useState, useEffect, useRef } from 'react';
import { useChat } from '../../context/ChatContext';
import { analysisApi, messageApi } from '../../api';

const ChatInput = () => {
  const [inputValue, setInputValue] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const { 
    activeChat, 
    addMessage, 
    dataFile, 
    suggestedQuestions, 
    setLoading, 
    setCurrentAnalysisResults,
    setCurrentQuery // Add the new function
  } = useChat();
  const textareaRef = useRef(null);
  
  // Adjust textarea height based on content
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = '44px'; // Reset height
      const scrollHeight = textareaRef.current.scrollHeight;
      textareaRef.current.style.height = `${Math.min(scrollHeight, 120)}px`;
    }
  }, [inputValue]);
  
  const handleInputChange = (e) => {
    setInputValue(e.target.value);
  };
  
  const handleSendMessage = async (e) => {
    if (e) e.preventDefault(); // Ensure preventDefault is called
    
    if (!inputValue.trim()) return;
    
    // Clear any previous error
    setErrorMessage('');
    
    // Log debug info
    console.log('Sending message:', inputValue);
    console.log('Active chat:', activeChat);
    console.log('Data file:', dataFile);
    
    // Create user message
    const userMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: inputValue,
      timestamp: new Date().toISOString(),
    };
    
    // Store the message value before clearing the input
    const messageContent = inputValue;
    
    // Save the current query before clearing the input
    setCurrentQuery(messageContent);
    
    // Clear input and start loading immediately
    addMessage(userMessage); // Add the user message to chat
    setInputValue('');
    setLoading(true);
    
    try {
      // Send the message to the backend
      let response;
      
      if (dataFile) {
        // If a data file is uploaded, send the message for analysis
        console.log('Analyzing data with file_id:', dataFile.id, 'and query:', messageContent);
        
        try {
          response = await analysisApi.analyzeData(dataFile.id, messageContent);
          console.log('Analysis response:', response);
        } catch (analyzeError) {
          console.error('Error during data analysis:', analyzeError);
          throw new Error(`Analysis request failed: ${analyzeError.message || 'Unknown error'}`);
        }
        
        // Add assistant message to the chat
        if (response && response.data && response.data.result) {
          const assistantMessage = {
            id: Date.now().toString(),
            role: 'assistant',
            content: response.data.result.text || response.data.result.explanation || 'Here is the analysis of your data.',
            timestamp: new Date().toISOString(),
          };
          
          // Add the assistant's response as a new message (doesn't replace user message)
          addMessage(assistantMessage);
          
          // If there are charts or visualizations in the response, add them as separate messages
          if (response.data.result.charts && response.data.result.charts.length > 0) {
            console.log('Adding chart messages:', response.data.result.charts.length);
            response.data.result.charts.forEach((chart) => {
              const chartMessage = {
                id: Date.now().toString(),
                role: 'assistant',
                type: 'chart',
                content: {
                  chartType: chart.type,
                  chartTitle: chart.title,
                  chartImage: chart.imageData,
                  chartId: chart.id
                },
                timestamp: new Date().toISOString(),
              };
              
              addMessage(chartMessage);
            });
          }
          
          // If there are tables, add them as separate messages
          if (response.data.result.tables && response.data.result.tables.length > 0) {
            response.data.result.tables.forEach((table) => {
              const tableMessage = {
                id: Date.now().toString(),
                role: 'assistant',
                type: 'table',
                content: {
                  tableData: table.data,
                  tableTitle: table.title,
                },
                timestamp: new Date().toISOString(),
              };
              
              addMessage(tableMessage);
            });
          }
          
          // Store the analysis results
          setCurrentAnalysisResults(response.data.result);
        } else {
          console.warn('Response or result missing:', response);
          const errorMessage = {
            id: Date.now().toString(),
            role: 'assistant',
            content: 'Sorry, I couldn\'t analyze your data. Please try a different question.',
            timestamp: new Date().toISOString(),
          };
          
          addMessage(errorMessage);
        }
      } else {
        // If no data file is uploaded, just send the message as a regular chat message
        console.log('Sending regular chat message to chat ID:', activeChat?.id);
        
        try {
          response = await messageApi.sendMessage(activeChat.id, {
            content: messageContent,
            role: 'user',
          });
          console.log('Message response:', response);
        } catch (messageError) {
          console.error('Error sending message:', messageError);
          throw new Error(`Message request failed: ${messageError.message || 'Unknown error'}`);
        }
        
        // Add assistant message to the chat
        if (response && response.data && response.data.message) {
          const assistantMessage = {
            id: Date.now().toString(),
            role: 'assistant',
            content: response.data.message.content,
            timestamp: new Date().toISOString(),
          };
          
          addMessage(assistantMessage);
        } else {
          console.warn('Message response invalid:', response);
          throw new Error('Invalid response from server');
        }
      }
    } catch (err) {
      console.error('Error in handleSendMessage:', err);
      setErrorMessage(err.message || 'An error occurred while processing your message');
      
      // Add error message to the chat
      const errorMessage = {
        id: Date.now().toString(),
        role: 'assistant',
        content: `Sorry, I encountered an error: ${err.response?.data?.message || err.message || 'Something went wrong. Please try again.'}`,
        timestamp: new Date().toISOString(),
      };
      
      addMessage(errorMessage);
    } finally {
      setLoading(false);
    }
  };
  
  const handleSuggestedQuestionClick = (question) => {
    setInputValue(question);
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  };
  
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault(); // Prevent default form submission behavior
      handleSendMessage();
    }
  };
  
  return (
    <div className="border-t border-gray-200 p-4">
      {errorMessage && (
        <div className="mb-4 p-3 bg-red-100 text-red-700 rounded-md">
          <p className="text-sm">{errorMessage}</p>
        </div>
      )}
      
      {suggestedQuestions && suggestedQuestions.length > 0 && (
        <div className="mb-4">
          <h3 className="text-sm font-medium text-gray-700 mb-2">Suggested questions:</h3>
          <div className="flex flex-wrap gap-2">
            {suggestedQuestions.map((question, index) => (
              <button
                key={index}
                onClick={() => handleSuggestedQuestionClick(question)}
                className="inline-flex items-center px-3 py-1 rounded-full text-sm bg-indigo-100 text-indigo-800 hover:bg-indigo-200 transition-colors"
              >
                {question}
              </button>
            ))}
          </div>
        </div>
      )}
      
      <form onSubmit={handleSendMessage} className="flex items-end gap-2">
        <div className="flex-1 relative">
          <textarea
            ref={textareaRef}
            value={inputValue}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            rows={1}
            className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent resize-none overflow-hidden"
            placeholder="Type your message or ask a question about your data..."
            style={{ minHeight: '44px', maxHeight: '120px' }}
          />
        </div>
        <button
          type="button" 
          onClick={handleSendMessage}
          className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 flex items-center justify-center"
          aria-label="Send message"
        >
          <svg 
            className="h-5 w-5" 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24" 
            xmlns="http://www.w3.org/2000/svg"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth="2" 
              d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
            />
          </svg>
        </button>
      </form>
    </div>
  );
};

export default ChatInput;