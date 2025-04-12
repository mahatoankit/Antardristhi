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
  const { messages, loading } = useChat();
  const messagesEndRef = useRef(null);

  // Automatically scroll to the latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Render the content of a message based on its type
  const renderMessageContent = (message) => {
    const { type, content } = message;

    if (type === 'error') {
      return (
        <p className="text-sm text-red-500">
          {content || 'An error occurred while processing your request.'}
        </p>
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
    } else {
      return (
        <p className="text-sm text-gray-500 italic">Unsupported message format.</p>
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