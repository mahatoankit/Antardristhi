import { useAuth } from '../../context/AuthContext';
import { useChat } from '../../context/ChatContext';
import ChatSidebar from './ChatSidebar';
import MessageList from './MessageList';
import ChatInput from './ChatInput';
import DataAnalysisPanel from '../dataAnalysis/DataAnalysisPanel';

const Chat = () => {
  const { user } = useAuth();
  const { dataFile } = useChat();
  
  return (
    <div className="h-screen flex overflow-hidden">
      {/* Chat sidebar for history */}
      <ChatSidebar />
      
      {/* Main chat area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <header className="bg-white border-b border-gray-200 px-4 py-2 flex items-center justify-between">
          <h1 className="text-xl font-semibold text-gray-900">
            Antardristhi Chat
          </h1>
          <div className="text-sm text-gray-500">
            <span className="capitalize">
              {user?.profile?.replace('_', ' ')} Profile
            </span>
          </div>
        </header>
        
        <div className="flex-1 flex overflow-hidden">
          {/* Messages and chat input */}
          <div className="flex-1 flex flex-col overflow-hidden">
            <MessageList />
            <ChatInput />
          </div>
          
          {/* Right panel for data analysis controls */}
          <div className="w-80 border-l border-gray-200 overflow-y-auto">
            <DataAnalysisPanel />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Chat;