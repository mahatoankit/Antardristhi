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

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const renderMessageContent = (message) => {
    const { type, content } = message;

    if (type === 'chart' && content.chartData) {
      return renderChart(content);
    } else if (type === 'table' && content.tableData) {
      return renderTable(content);
    } else if (type === 'file-info' && content.fileInfo) {
      return renderFileInfo(content.fileInfo);
    } else if (typeof content === 'string') {
      return (
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            p: ({ children }) => (
              <p className="prose prose-sm max-w-none">{children}</p>
            ),
            code: ({ children }) => (
              <code className="bg-gray-700 text-white px-1 py-0.5 rounded">
                {children}
              </code>
            ),
            pre: ({ children }) => (
              <pre className="bg-gray-700 text-white overflow-x-auto p-2 rounded">
                {children}
              </pre>
            ),
          }}
        >
          {content}
        </ReactMarkdown>
      );
    } else if (typeof content === 'object' && content !== null) {
      return (
        <pre className="bg-gray-100 p-2 text-xs rounded overflow-x-auto text-red-500">
          {JSON.stringify(content, null, 2)}
        </pre>
      );
    } else {
      return (
        <p className="text-sm text-gray-500 italic">Unsupported message format.</p>
      );
    }
  };

  const renderChart = (content) => {
    const { chartData, chartTitle } = content;

    return (
      <div className="w-full">
        <h4 className="text-sm font-semibold mb-2">{chartTitle}</h4>
        <div className="bg-gray-50 p-2 rounded-md">
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
      </div>
    );
  };

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
                  scope="col"
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

  const renderFileInfo = (fileInfo) => (
    <div className="flex items-center p-4 bg-gray-50 rounded-md">
      <div className="flex-shrink-0 h-10 w-10 bg-indigo-100 rounded-md flex items-center justify-center">
        <svg
          className="h-6 w-6 text-indigo-600"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
          />
        </svg>
      </div>
      <div className="ml-4">
        <h4 className="text-sm font-medium text-gray-900">{fileInfo.name}</h4>
        <p className="text-xs text-gray-500">{formatFileSize(fileInfo.size)}</p>
        {fileInfo.summary && (
          <p className="text-xs text-gray-700 mt-1">{fileInfo.summary}</p>
        )}
      </div>
    </div>
  );

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    else return (bytes / 1048576).toFixed(1) + ' MB';
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
            <div
              className={`text-sm ${
                message.role === 'user' ? 'text-white' : 'text-gray-900'
              }`}
            >
              {renderMessageContent(message)}
            </div>
          </div>
        </div>
      ))}

      {loading && (
        <div className="flex justify-start">
          <div className="max-w-3xl rounded-lg px-4 py-2 bg-white border border-gray-200">
            <div className="flex space-x-2 items-center">
              <div
                className="w-2 h-2 rounded-full bg-gray-400 animate-bounce"
                style={{ animationDelay: '0ms' }}
              ></div>
              <div
                className="w-2 h-2 rounded-full bg-gray-400 animate-bounce"
                style={{ animationDelay: '300ms' }}
              ></div>
              <div
                className="w-2 h-2 rounded-full bg-gray-400 animate-bounce"
                style={{ animationDelay: '600ms' }}
              ></div>
            </div>
          </div>
        </div>
      )}

      <div ref={messagesEndRef} />
    </div>
  );
};

export default MessageList;
