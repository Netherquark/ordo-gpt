import { Plus, Settings, LogOut, User, Sparkles } from 'lucide-react';
import { Button } from './ui/button';
import { ScrollArea } from './ui/scroll-area';
import type { Conversation, User as UserType } from '../App';

interface SidebarProps {
  conversations: Conversation[];
  currentConversationId: string | null;
  onSelectConversation: (id: string) => void;
  onNewConversation: () => void;
  onOpenApiConfig: () => void;
  user: UserType;
  onLogout: () => void;
}

export function Sidebar({
  conversations,
  currentConversationId,
  onSelectConversation,
  onNewConversation,
  onOpenApiConfig,
  user,
  onLogout
}: SidebarProps) {
  const formatTime = (timestamp: number) => {
    const diff = Date.now() - timestamp;
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    return `${days}d ago`;
  };

  return (
    <div className="w-64 bg-[#0f0f0f] border-r border-[#1a1a1a] flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-[#1a1a1a] flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-[#10b981] flex items-center justify-center">
            <Sparkles className="h-4 w-4 text-white" />
          </div>
          <span className="text-white">HNet AI</span>
        </div>
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8 text-gray-400 hover:text-white hover:bg-[#1a1a1a]"
          onClick={onOpenApiConfig}
        >
          <Settings className="h-4 w-4" />
        </Button>
      </div>

      {/* New Conversation Button */}
      <div className="p-4">
        <Button
          onClick={onNewConversation}
          className="w-full bg-[#10b981] hover:bg-[#059669] text-white gap-2"
        >
          <Plus className="h-4 w-4" />
          New Conversation
        </Button>
      </div>

      {/* Recent Conversations */}
      <div className="px-4 pb-2">
        <h3 className="text-xs text-gray-500">Recent Conversations</h3>
      </div>

      <ScrollArea className="flex-1 px-2">
        <div className="space-y-1">
          {conversations.map((conv) => (
            <button
              key={conv.id}
              onClick={() => onSelectConversation(conv.id)}
              className={`w-full p-3 rounded-lg text-left transition-colors ${
                currentConversationId === conv.id
                  ? 'bg-[#1a1a1a]'
                  : 'hover:bg-[#1a1a1a]'
              }`}
            >
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 rounded-full bg-[#10b981] flex items-center justify-center flex-shrink-0 mt-0.5">
                  <span className="text-xs">H</span>
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-white truncate">{conv.title}</p>
                  <p className="text-xs text-gray-500 flex items-center gap-1">
                    <span className="inline-block w-1 h-1 rounded-full bg-gray-500"></span>
                    {formatTime(conv.timestamp)}
                  </p>
                </div>
              </div>
            </button>
          ))}
        </div>
      </ScrollArea>

      {/* Footer */}
      <div className="p-4 border-t border-[#1a1a1a] space-y-3">
        <div className="flex items-center gap-3 p-2 bg-[#1a1a1a] rounded-lg">
          <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center flex-shrink-0">
            <User className="h-4 w-4" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm text-white truncate">{user.username}</p>
            <p className="text-xs text-gray-500">Connected</p>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={onLogout}
            className="h-8 w-8 text-gray-400 hover:text-white hover:bg-[#2a2a2a]"
          >
            <LogOut className="h-4 w-4" />
          </Button>
        </div>
        <div className="text-center">
          <h4 className="text-sm text-[#10b981]">HNet v1.0</h4>
          <p className="text-xs text-gray-500 mt-1">
            AI Language Model Platform
          </p>
        </div>
      </div>
    </div>
  );
}
