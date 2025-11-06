import { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatArea } from './components/ChatArea';
import { ApiConfigDialog } from './components/ApiConfigDialog';
import { LoginPage } from './components/LoginPage';

export interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp?: number;
}

export interface Conversation {
  id: string;
  title: string;
  timestamp: number;
  messages: Message[];
}

export interface User {
  username: string;
  apiKey: string;
}

export default function App() {
  const [user, setUser] = useState<User | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([
    {
      id: '1',
      title: 'Generate creative product descriptions',
      timestamp: Date.now() - 1200000,
      messages: []
    },
    {
      id: '2',
      title: 'Code review and optimization tips',
      timestamp: Date.now() - 3600000,
      messages: []
    },
    {
      id: '3',
      title: 'Data analysis and insights',
      timestamp: Date.now() - 7200000,
      messages: []
    },
    {
      id: '4',
      title: 'Technical documentation writing',
      timestamp: Date.now() - 21600000,
      messages: []
    },
    {
      id: '5',
      title: 'API integration strategies',
      timestamp: Date.now() - 86400000,
      messages: []
    }
  ]);
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);
  const [apiEndpoint, setApiEndpoint] = useState('http://192.168.1.137:5000');
  const [showApiConfig, setShowApiConfig] = useState(false);

  // Load user from localStorage on mount
  useEffect(() => {
    const savedUser = localStorage.getItem('hnet_user');
    const savedEndpoint = localStorage.getItem('hnet_api_endpoint');
    if (savedUser) {
      setUser(JSON.parse(savedUser));
    }
    if (savedEndpoint) {
      setApiEndpoint(savedEndpoint);
    }
  }, []);

  const handleLogin = (username: string, apiKey: string) => {
    const newUser = { username, apiKey };
    setUser(newUser);
    localStorage.setItem('hnet_user', JSON.stringify(newUser));
  };

  const handleLogout = () => {
    setUser(null);
    localStorage.removeItem('hnet_user');
    setConversations([
      {
        id: '1',
        title: 'Generate creative product descriptions',
        timestamp: Date.now() - 1200000,
        messages: []
      },
      {
        id: '2',
        title: 'Code review and optimization tips',
        timestamp: Date.now() - 3600000,
        messages: []
      },
      {
        id: '3',
        title: 'Data analysis and insights',
        timestamp: Date.now() - 7200000,
        messages: []
      },
      {
        id: '4',
        title: 'Technical documentation writing',
        timestamp: Date.now() - 21600000,
        messages: []
      },
      {
        id: '5',
        title: 'API integration strategies',
        timestamp: Date.now() - 86400000,
        messages: []
      }
    ]);
    setCurrentConversationId(null);
  };

  const createNewConversation = () => {
    const newConv: Conversation = {
      id: Date.now().toString(),
      title: 'New conversation',
      timestamp: Date.now(),
      messages: []
    };
    setConversations([newConv, ...conversations]);
    setCurrentConversationId(newConv.id);
  };

  const updateConversation = (id: string, messages: Message[]) => {
    setConversations(convs =>
      convs.map(c => {
        if (c.id === id) {
          // Update title based on first message if it's still "New conversation"
          let title = c.title;
          if (title === 'New conversation' && messages.length > 0 && messages[0].role === 'user') {
            title = messages[0].content.slice(0, 40) + (messages[0].content.length > 40 ? '...' : '');
          }
          return { ...c, messages, title, timestamp: Date.now() };
        }
        return c;
      })
    );
  };

  const handleSaveApiEndpoint = (endpoint: string) => {
    setApiEndpoint(endpoint);
    localStorage.setItem('hnet_api_endpoint', endpoint);
  };

  const currentConversation = conversations.find(c => c.id === currentConversationId);

  if (!user) {
    return <LoginPage onLogin={handleLogin} />;
  }

  return (
    <div className="flex h-screen bg-[#0a0a0a] text-white">
      <Sidebar
        conversations={conversations}
        currentConversationId={currentConversationId}
        onSelectConversation={setCurrentConversationId}
        onNewConversation={createNewConversation}
        onOpenApiConfig={() => setShowApiConfig(true)}
        user={user}
        onLogout={handleLogout}
      />
      <ChatArea
        conversation={currentConversation}
        apiEndpoint={apiEndpoint}
        onUpdateMessages={(messages) => {
          if (currentConversationId) {
            updateConversation(currentConversationId, messages);
          }
        }}
      />
      <ApiConfigDialog
        open={showApiConfig}
        onOpenChange={setShowApiConfig}
        apiEndpoint={apiEndpoint}
        onSave={handleSaveApiEndpoint}
      />
    </div>
  );
}
