import { useState } from 'react';
import { useChat } from '../../context/ChatContext';
import FileUpload from '../fileUpload/FileUpload';
import { analysisApi } from '../../api';

const DataAnalysisPanel = () => {
  const { dataFile, analysisResults, setCurrentAnalysisResults, setLoading, addMessage } = useChat();
  const [showDataPreview, setShowDataPreview] = useState(false);
  const [dataPreview, setDataPreview] = useState(null);
  
  const handleGetDataPreview = async () => {
    if (!dataFile) return;
    
    setLoading(true);
    try {
      const response = await analysisApi.getDataPreview(dataFile.id);
      setDataPreview(response.data.preview);
      setShowDataPreview(true);
    } catch (err) {
      console.error('Error getting data preview:', err);
      // Add error message to the chat
      const errorMessage = {
        id: Date.now().toString(),
        role: 'assistant',
        content: `Sorry, I couldn't generate a preview of your data: ${err.response?.data?.message || 'Something went wrong.'}`,
        timestamp: new Date().toISOString(),
      };
      
      addMessage(errorMessage);
    } finally {
      setLoading(false);
    }
  };
  
  const handleDataCleanlinessCheck = async () => {
    if (!dataFile) return;
    
    setLoading(true);
    
    // Add user message asking for data cleanliness check
    const userMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: 'Check the cleanliness of this data and identify any issues.',
      timestamp: new Date().toISOString(),
    };
    
    addMessage(userMessage);
    
    try {
      const response = await analysisApi.analyzeData(dataFile.id, 'Check the cleanliness of this data and identify any issues or potential preprocessing steps needed.');
      
      // Add assistant message with cleanliness report
      if (response.data.result) {
        const assistantMessage = {
          id: Date.now().toString(),
          role: 'assistant',
          content: response.data.result.text || 'Here is my analysis of your data cleanliness.',
          timestamp: new Date().toISOString(),
        };
        
        addMessage(assistantMessage);
        setCurrentAnalysisResults(response.data.result);
      }
    } catch (err) {
      // Add error message to the chat
      const errorMessage = {
        id: Date.now().toString(),
        role: 'assistant',
        content: `Sorry, I couldn't analyze your data cleanliness: ${err.response?.data?.message || 'Something went wrong.'}`,
        timestamp: new Date().toISOString(),
      };
      
      addMessage(errorMessage);
    } finally {
      setLoading(false);
    }
  };
  
  // If no data file is uploaded, show the file upload component
  if (!dataFile) {
    return (
      <div className="p-4 bg-white rounded-lg shadow">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Upload a Dataset for Analysis</h3>
        <p className="text-sm text-gray-600 mb-4">
          Upload a CSV, JSON, or XLSX file (up to 20MB) to get started with data analysis.
        </p>
        <FileUpload />
      </div>
    );
  }
  
  return (
    <div className="p-4 bg-white rounded-lg shadow">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-medium text-gray-900">Dataset Analysis</h3>
        <button
          onClick={() => setCurrentAnalysisResults(null)}
          className="text-sm text-gray-600 hover:text-gray-900"
        >
          Upload a different file
        </button>
      </div>
      
      <div className="flex items-center mb-4 p-3 bg-gray-50 rounded-lg">
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
          <h4 className="text-sm font-medium text-gray-900">{dataFile.name}</h4>
          <p className="text-xs text-gray-500">
            {dataFile.size < 1024
              ? dataFile.size + ' B'
              : dataFile.size < 1048576
              ? (dataFile.size / 1024).toFixed(1) + ' KB'
              : (dataFile.size / 1048576).toFixed(1) + ' MB'}
          </p>
        </div>
      </div>
      
      <div className="flex space-x-2 mb-4">
        <button
          onClick={handleGetDataPreview}
          className="px-3 py-1 text-sm border border-gray-300 rounded-md hover:bg-gray-50"
        >
          Preview Data
        </button>
        <button
          onClick={handleDataCleanlinessCheck}
          className="px-3 py-1 text-sm border border-gray-300 rounded-md hover:bg-gray-50"
        >
          Check Data Cleanliness
        </button>
      </div>
      
      {showDataPreview && dataPreview && (
        <div className="mt-4">
          <div className="flex justify-between items-center mb-2">
            <h4 className="text-sm font-medium text-gray-900">Data Preview</h4>
            <button
              onClick={() => setShowDataPreview(false)}
              className="text-sm text-gray-600 hover:text-gray-900"
            >
              Close
            </button>
          </div>
          <div className="border border-gray-200 rounded-md overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  {dataPreview.columns.map((column, idx) => (
                    <th
                      key={idx}
                      scope="col"
                      className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                    >
                      {column}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {dataPreview.data.map((row, rowIdx) => (
                  <tr key={rowIdx}>
                    {row.map((cell, cellIdx) => (
                      <td key={cellIdx} className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {cell}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            Showing {dataPreview.data.length} of {dataPreview.total_rows} rows.
          </p>
        </div>
      )}
    </div>
  );
};

export default DataAnalysisPanel;