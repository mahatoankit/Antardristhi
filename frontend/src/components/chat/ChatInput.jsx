import { useState } from 'react';
import { useChat } from '../../context/ChatContext';
import { analysisApi, messageApi } from '../../api';

const ChatInput = () => {
  const [inputValue, setInputValue] = useState('');
  const { activeChat, addMessage, dataFile, suggestedQuestions, setLoading, setCurrentAnalysisResults } = useChat();
  
  const handleInputChange = (e) => {
    setInputValue(e.target.value);
  };
  
  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputValue.trim()) return;
    
    // Add user message to the chat
    const userMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: inputValue,
      timestamp: new Date().toISOString(),
    };
    
    addMessage(userMessage);
    setInputValue('');
    setLoading(true);
    
    try {
      // Send the message to the backend
      let response;
      
      if (dataFile) {
        // If a data file is uploaded, send the message for analysis
        response = await analysisApi.analyzeData(dataFile.id, inputValue);
        
        // Add assistant message to the chat
        if (response.data.result) {
          const assistantMessage = {
            id: Date.now().toString(),
            role: 'assistant',
            content: response.data.result.text || 'Here is the analysis of your data.',
            timestamp: new Date().toISOString(),
          };
          
          addMessage(assistantMessage);
          
          // If there are charts or visualizations, add them as separate messages
          if (response.data.result.charts && response.data.result.charts.length > 0) {
            response.data.result.charts.forEach((chart) => {
              const chartMessage = {
                id: Date.now().toString(),
                role: 'assistant',
                type: 'chart',
                content: {
                  chartData: chart.data,
                  chartTitle: chart.title,
                  chartType: chart.type,
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
        }
      } else {
        // If no data file is uploaded, just send the message as a regular chat message
        response = await messageApi.sendMessage(activeChat.id, {
          content: inputValue,
          role: 'user',
        });
        
        // Add assistant message to the chat
        if (response.data.message) {
          const assistantMessage = {
            id: Date.now().toString(),
            role: 'assistant',
            content: response.data.message.content,
            timestamp: new Date().toISOString(),
          };
          
          addMessage(assistantMessage);
        }
      }
    } catch (err) {
      // Add error message to the chat
      const errorMessage = {
        id: Date.now().toString(),
        role: 'assistant',
        content: `Sorry, I encountered an error: ${err.response?.data?.message || 'Something went wrong. Please try again.'}`,
        timestamp: new Date().toISOString(),
      };
      
      addMessage(errorMessage);
    } finally {
      setLoading(false);
    }
  };
  
  const handleSuggestedQuestionClick = (question) => {
    setInputValue(question);
  };
  
  return (
    <div className="border-t border-gray-200 p-4">
      {suggestedQuestions.length > 0 && (
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
            value={inputValue}
            onChange={handleInputChange}
            rows={1}
            className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent resize-none overflow-hidden"
            placeholder="Type your message or ask a question about your data..."
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSendMessage(e);
              }
            }}
            style={{ minHeight: '44px', maxHeight: '120px' }}
          />
          <div className="text-xs text-gray-500 mt-1">
            Press Enter to send, Shift+Enter for a new line
          </div>
        </div>
        <button
          type="submit"
          className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
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
              strokeWidth={2}
              d="M14 5l7 7m0 0l-7 7m7-7H3"
            />
          </svg>
        </button>
      </form>
    </div>
  );
};

export default ChatInput;