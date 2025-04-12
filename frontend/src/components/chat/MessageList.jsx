import { useRef, useEffect } from 'react';
import { useChat } from '../../context/ChatContext';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

const MessageList = () => {
  const { messages, loading, lastQuery, analysisResults } = useChat();
  const messagesEndRef = useRef(null);

  // Automatically scroll to the latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Render the content of a message based on its type
  const renderMessageContent = (message) => {
    const { type, content, role } = message;
    
    // For assistant messages that contain analysis results, show the query that generated them
    if (role === 'assistant' && analysisResults && lastQuery && 
        (message.content === analysisResults.text || message.content === analysisResults.explanation)) {
      return (
        <div>
          <div className="mb-2 text-xs text-gray-500 italic border-b border-gray-200 pb-1">
            In response to: "{lastQuery}"
          </div>
          {renderRegularMessage(message)}
        </div>
      );
    }
    
    return renderRegularMessage(message);
  };
  
  // Render regular message content
  const renderRegularMessage = (message) => {
    const { type, content } = message;

    if (type === 'error') {
      return (
        <p className="text-sm text-red-500">
          {content || 'An error occurred while processing your request.'}
        </p>
      );
    } else if (type === 'chart' && content.chartImage) {
      // Handle chart messages with image data
      return (
        <div className="w-full">
          <h4 className="text-sm font-semibold mb-2">{content.chartTitle || 'Chart'}</h4>
          <img 
            src={content.chartImage.startsWith('data:image') ? content.chartImage : `data:image/png;base64,${content.chartImage}`} 
            alt={content.chartTitle || 'Chart'} 
            className="max-w-full h-auto rounded-md"
          />
        </div>
      );
    } else if (type === 'chart' && content.chartData) {
      return renderChart(content);
    } else if (type === 'table' && content.tableData) {
      return renderTable(content);
    } else if (typeof content === 'string') {
      return (
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            p: ({ children }) => (
              <p className="prose prose-sm max-w-none">{children}</p>
            ),
          }}
        >
          {content}
        </ReactMarkdown>
      );
    } else if (content && typeof content === 'object') {
      // Handle complex object content
      if (content.tableData) {
        return renderTable(content);
      } else if (content.chartData || content.chartImage || content.imageData) {
        // Support both chartImage and the new imageData field
        if (content.imageData) {
          return (
            <div className="w-full">
              <h4 className="text-sm font-semibold mb-2">{content.chartTitle || content.title || 'Visualization'}</h4>
              <img 
                src={content.imageData} 
                alt={content.chartTitle || content.title || 'Visualization'} 
                className="max-w-full h-auto rounded-md shadow-sm border border-gray-200"
              />
            </div>
          );
        }
        return renderChart(content);
      } else if (content.text || content.explanation) {
        // Handle text/explanation content
        return (
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              p: ({ children }) => (
                <p className="prose prose-sm max-w-none">{children}</p>
              ),
            }}
          >
            {content.text || content.explanation}
          </ReactMarkdown>
        );
      } else {
        // Render any other object as JSON
        return (
          <div className="w-full overflow-x-auto">
            <pre className="text-xs bg-gray-100 p-2 rounded">
              {JSON.stringify(content, null, 2)}
            </pre>
          </div>
        );
      }
    } else {
      return (
        <p className="text-sm text-gray-500 italic">Empty or unsupported message format.</p>
      );
    }
  };

  // Render a chart message
  const renderChart = (content) => {
    const { chartData, chartTitle } = content;

    return (
      <div className="w-full">
        <h4 className="text-sm font-semibold mb-2">{chartTitle}</h4>
        <ResponsiveContainer width="100%" height={250}>
          <AreaChart
            data={chartData}
            margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Area
              type="monotone"
              dataKey="value"
              stroke="#8884d8"
              fill="#8884d8"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    );
  };

  // Render a table message
  const renderTable = (content) => {
    const { tableData, tableTitle } = content;

    if (!tableData || !tableData.length) return null;

    const headers = Object.keys(tableData[0]);

    return (
      <div className="w-full overflow-x-auto">
        <h4 className="text-sm font-semibold mb-2">{tableTitle}</h4>
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              {headers.map((header) => (
                <th
                  key={header}
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                >
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {tableData.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {headers.map((header) => (
                  <td
                    key={`${rowIndex}-${header}`}
                    className="px-6 py-4 whitespace-nowrap text-sm text-gray-500"
                  >
                    {row[header]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4">
      {messages.map((message, index) => (
        <div
          key={index}
          className={`flex ${
            message.role === 'user' ? 'justify-end' : 'justify-start'
          }`}
        >
          <div
            className={`max-w-3xl rounded-lg px-4 py-2 ${
              message.role === 'user'
                ? 'bg-indigo-600 text-white'
                : 'bg-white border border-gray-200'
            }`}
          >
            {renderMessageContent(message)}
          </div>
        </div>
      ))}

      {loading && (
        <div className="flex justify-start">
          <div className="max-w-3xl rounded-lg px-4 py-2 bg-white border border-gray-200">
            <div className="flex space-x-2 items-center">
              <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce"></div>
              <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce"></div>
              <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce"></div>
            </div>
          </div>
        </div>
      )}

      <div ref={messagesEndRef} />
    </div>
  );
};

export default MessageList;