import { useState } from 'react';
import { useChat } from '../../context/ChatContext';
import { useAuth, USER_PROFILES } from '../../context/AuthContext';

const ChatSidebar = () => {
  const { chats, activeChat, createNewChat, switchChat, deleteChat, renameChat } = useChat();
  const { user, logout, changeProfile } = useAuth();
  const [editingChatId, setEditingChatId] = useState(null);
  const [newTitle, setNewTitle] = useState('');
  const [showProfileMenu, setShowProfileMenu] = useState(false);

  const handleCreateNewChat = () => {
    createNewChat();
  };

  const handleSwitchChat = (chatId) => {
    switchChat(chatId);
  };

  const handleDeleteChat = (e, chatId) => {
    e.stopPropagation();
    deleteChat(chatId);
  };

  const handleRenameChat = (e, chatId, chatTitle) => {
    e.stopPropagation();
    setEditingChatId(chatId);
    setNewTitle(chatTitle);
  };

  const submitRename = (e) => {
    e.preventDefault();
    if (newTitle.trim()) {
      renameChat(editingChatId, newTitle.trim());
      setEditingChatId(null);
      setNewTitle('');
    }
  };

  const handleChangeProfile = (profile) => {
    changeProfile(profile);
    setShowProfileMenu(false);
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
    }).format(date);
  };

  return (
    <div className="h-full flex flex-col bg-gray-800 text-white w-64">
      <div className="p-4 border-b border-gray-700">
        <button
          onClick={handleCreateNewChat}
          className="w-full flex items-center justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
        >
          <svg
            className="h-4 w-4 mr-2"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 6v6m0 0v6m0-6h6m-6 0H6"
            />
          </svg>
          New Chat
        </button>
      </div>

      <div className="flex-1 overflow-y-auto pb-4">
        <div className="px-3 py-2 text-xs text-gray-400 uppercase">Chats</div>
        <ul>
          {chats.map((chat) => (
            <li
              key={chat.id}
              className={`px-3 py-2 cursor-pointer hover:bg-gray-700 ${
                activeChat?.id === chat.id ? 'bg-gray-700' : ''
              }`}
            >
              {editingChatId === chat.id ? (
                <form onSubmit={submitRename} className="flex">
                  <input
                    type="text"
                    value={newTitle}
                    onChange={(e) => setNewTitle(e.target.value)}
                    className="flex-1 bg-gray-900 text-white text-sm rounded px-2 py-1"
                    autoFocus
                    onBlur={submitRename}
                  />
                </form>
              ) : (
                <div
                  className="flex items-start justify-between"
                  onClick={() => handleSwitchChat(chat.id)}
                >
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium truncate">{chat.title}</div>
                    <div className="text-xs text-gray-400">
                      {formatDate(chat.createdAt)}
                    </div>
                  </div>
                  <div className="flex space-x-1">
                    <button
                      onClick={(e) => handleRenameChat(e, chat.id, chat.title)}
                      className="text-gray-400 hover:text-white"
                    >
                      <svg
                        className="h-4 w-4"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                        xmlns="http://www.w3.org/2000/svg"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z"
                        />
                      </svg>
                    </button>
                    <button
                      onClick={(e) => handleDeleteChat(e, chat.id)}
                      className="text-gray-400 hover:text-red-500"
                    >
                      <svg
                        className="h-4 w-4"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                        xmlns="http://www.w3.org/2000/svg"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                        />
                      </svg>
                    </button>
                  </div>
                </div>
              )}
            </li>
          ))}
        </ul>
      </div>

      <div className="p-4 border-t border-gray-700">
        <div className="relative">
          <button
            onClick={() => setShowProfileMenu(!showProfileMenu)}
            className="flex items-center justify-between w-full px-4 py-2 text-sm text-left font-medium rounded-md bg-gray-700 hover:bg-gray-600 focus:outline-none"
          >
            <div className="flex items-center">
              <div className="flex-shrink-0 h-8 w-8 rounded-full bg-indigo-500 flex items-center justify-center">
                {user?.name?.charAt(0) || 'U'}
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium text-white truncate">
                  {user?.name || 'User'}
                </p>
                <p className="text-xs font-medium text-gray-300 capitalize">
                  {user?.profile?.replace('_', ' ') || 'No Profile'}
                </p>
              </div>
            </div>
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
                d="M19 9l-7 7-7-7"
              />
            </svg>
          </button>

          {showProfileMenu && (
            <div className="absolute bottom-full left-0 right-0 mb-2 rounded-md shadow-lg bg-gray-700 ring-1 ring-black ring-opacity-5 z-10">
              <div className="py-1">
                <div className="px-4 py-2 text-xs text-gray-400">
                  Change Profile
                </div>
                {Object.entries(USER_PROFILES).map(([key, value]) => (
                  <button
                    key={key}
                    onClick={() => handleChangeProfile(value)}
                    className={`block w-full px-4 py-2 text-sm text-left hover:bg-gray-600 ${
                      user?.profile === value ? 'bg-gray-600' : ''
                    }`}
                  >
                    <span className="capitalize">{value.replace('_', ' ')}</span>
                  </button>
                ))}
                <div className="border-t border-gray-600 my-1"></div>
                <button
                  onClick={logout}
                  className="block w-full px-4 py-2 text-sm text-left text-red-400 hover:bg-gray-600"
                >
                  Sign out
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChatSidebar;