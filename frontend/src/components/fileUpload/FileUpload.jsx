import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { analysisApi } from '../../api';
import { useChat } from '../../context/ChatContext';

const FileUpload = () => {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const { activeChat, setCurrentDataFile, setLoading, setCurrentSuggestedQuestions } = useChat();
  
  const onDrop = useCallback(async (acceptedFiles) => {
    // Check file size (20MB limit)
    const maxSize = 20 * 1024 * 1024; // 20MB in bytes
    const file = acceptedFiles[0];
    
    if (file.size > maxSize) {
      setError('File size exceeds the 20MB limit.');
      return;
    }
    
    // Check file type
    const validTypes = ['text/csv', 'application/json', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'];
    if (!validTypes.includes(file.type)) {
      setError('Invalid file type. Please upload a CSV, JSON, or XLSX file.');
      return;
    }
    
    setUploading(true);
    setError('');
    
    try {
      // Upload the file to the backend
      const response = await analysisApi.uploadFile(file, activeChat?.id);
      
      // Get file ID from the appropriate location in the response
      const fileId = response.data.ml_preprocessing?.data_id;
      
      if (!fileId) {
        throw new Error('File ID not found in the server response');
      }
      
      // Update the file in the current chat
      setCurrentDataFile({
        id: fileId,
        name: file.name,
        type: file.type,
        size: file.size,
        uploadedAt: new Date().toISOString(),
      });
      
      // Get suggested questions based on the uploaded file
      setLoading(true);
      const suggestionsResponse = await analysisApi.getSuggestedQuestions(fileId);
      setCurrentSuggestedQuestions(suggestionsResponse.data.questions || []);
      
      setUploading(false);
      setLoading(false);
    } catch (err) {
      setError(err.response?.data?.message || err.message || 'An error occurred while uploading the file.');
      setUploading(false);
    }
  }, [activeChat, setCurrentDataFile, setLoading, setCurrentSuggestedQuestions]);
  
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/json': ['.json'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
    },
    maxFiles: 1,
  });
  
  return (
    <div className="w-full">
      <div
        {...getRootProps()}
        className={`p-6 border-2 border-dashed rounded-lg text-center cursor-pointer transition-colors ${
          isDragActive ? 'border-indigo-500 bg-indigo-50' : 'border-gray-300 hover:border-indigo-400'
        }`}
      >
        <input {...getInputProps()} />
        {uploading ? (
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-indigo-500 mx-auto"></div>
            <p className="mt-2 text-sm text-gray-600">Uploading...</p>
          </div>
        ) : (
          <div>
            <svg
              className="mx-auto h-12 w-12 text-gray-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
              />
            </svg>
            <p className="mt-2 text-sm text-gray-600">
              {isDragActive
                ? 'Drop the file here...'
                : 'Drag & drop a file here, or click to select a file'}
            </p>
            <p className="mt-1 text-xs text-gray-500">
              Supported formats: CSV, JSON, XLSX (Max size: 20MB)
            </p>
          </div>
        )}
      </div>
      
      {error && (
        <div className="mt-2 text-sm text-red-600">
          {error}
        </div>
      )}
    </div>
  );
};

export default FileUpload;