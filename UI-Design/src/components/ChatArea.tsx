import { useState, useEffect, useRef } from 'react';
import { Send, FileText, Code, Database, MessageSquare, Lightbulb, Workflow } from 'lucide-react';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';
import { ScrollArea } from './ui/scroll-area';
import { Badge } from './ui/badge';
import type { Conversation, Message } from '../App';

interface ChatAreaProps {
  conversation: Conversation | undefined;
  apiEndpoint: string;
  onUpdateMessages: (messages: Message[]) => void;
}

export function ChatArea({ conversation, apiEndpoint, onUpdateMessages }: ChatAreaProps) {
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  const messages = conversation?.messages || [];

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || !conversation) return;

    const userMessage: Message = {
      role: 'user',
      content: input,
      timestamp: Date.now()
    };

    const updatedMessages = [...messages, userMessage];
    onUpdateMessages(updatedMessages);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch(`${apiEndpoint}/v1/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'hnet-1stage-L',
          messages: updatedMessages.map(m => ({
            role: m.role,
            content: m.content
          })),
          max_tokens: 512,
          temperature: 0.8,
          top_p: 0.95
        })
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      const assistantMessage: Message = {
        role: 'assistant',
        content: data.choices?.[0]?.message?.content || 'No response',
        timestamp: Date.now()
      };

      onUpdateMessages([...updatedMessages, assistantMessage]);
    } catch (error) {
      console.error('Error calling API:', error);
      const errorMessage: Message = {
        role: 'assistant',
        content: `Error: ${error instanceof Error ? error.message : 'Failed to get response'}. Make sure the HNet server is running at ${apiEndpoint}`,
        timestamp: Date.now()
      };
      onUpdateMessages([...updatedMessages, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const suggestedPrompts = [
    { icon: FileText, text: 'Write a product description for eco-friendly water bottle' },
    { icon: Code, text: 'Review this Python code and suggest optimizations' },
    { icon: Database, text: 'Analyze sales data and identify trends' },
    { icon: MessageSquare, text: 'Generate email response to customer inquiry' },
    { icon: Lightbulb, text: 'Brainstorm marketing campaign ideas' },
    { icon: Workflow, text: 'Create a project timeline for app development' }
  ];

  if (!conversation) {
    return (
      <div className="flex-1 flex items-center justify-center flex-col gap-8 p-8">
        <div className="text-center space-y-6 max-w-2xl">
          <div className="w-20 h-20 bg-[#10b981] rounded-2xl flex items-center justify-center mx-auto">
            <MessageSquare className="w-12 h-12 text-white" />
          </div>
          <h1 className="text-3xl text-[#10b981]">HNet AI Platform</h1>
          <p className="text-gray-400">
            Powerful AI language model for content generation, code assistance, and data analysis
          </p>
          <div className="flex items-center justify-center gap-4 flex-wrap">
            <Badge variant="secondary" className="bg-[#1a1a1a] text-gray-300 border-[#2a2a2a]">
              <FileText className="w-3 h-3 mr-1 text-[#10b981]" />
              Content Writing
            </Badge>
            <Badge variant="secondary" className="bg-[#1a1a1a] text-gray-300 border-[#2a2a2a]">
              <Code className="w-3 h-3 mr-1 text-[#10b981]" />
              Code Analysis
            </Badge>
            <Badge variant="secondary" className="bg-[#1a1a1a] text-gray-300 border-[#2a2a2a]">
              <Database className="w-3 h-3 mr-1 text-[#10b981]" />
              Data Insights
            </Badge>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col">
      {/* Messages Area */}
      <ScrollArea className="flex-1 p-8" ref={scrollRef}>
        <div className="max-w-4xl mx-auto space-y-6">
          {messages.length === 0 ? (
            <div className="text-center space-y-6 py-12">
              <div className="w-20 h-20 bg-[#10b981] rounded-2xl flex items-center justify-center mx-auto">
                <MessageSquare className="w-12 h-12 text-white" />
              </div>
              <h1 className="text-3xl text-[#10b981]">Start a Conversation</h1>
              <p className="text-gray-400">
                Ask me anything - from writing content to analyzing code
              </p>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div key={idx} className="flex gap-4">
                {msg.role === 'user' ? (
                  <>
                    <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center flex-shrink-0">
                      <span className="text-sm">U</span>
                    </div>
                    <div className="flex-1">
                      <p className="text-white whitespace-pre-wrap">{msg.content}</p>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="w-8 h-8 rounded-full bg-[#10b981] flex items-center justify-center flex-shrink-0">
                      <span className="text-sm">H</span>
                    </div>
                    <div className="flex-1">
                      <p className="text-white whitespace-pre-wrap">{msg.content}</p>
                    </div>
                  </>
                )}
              </div>
            ))
          )}
          {isLoading && (
            <div className="flex gap-4">
              <div className="w-8 h-8 rounded-full bg-[#10b981] flex items-center justify-center flex-shrink-0">
                <span className="text-sm">H</span>
              </div>
              <div className="flex-1">
                <div className="flex gap-1">
                  <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                  <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                  <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                </div>
              </div>
            </div>
          )}
        </div>
      </ScrollArea>

      {/* Input Area */}
      <div className="border-t border-[#1a1a1a] p-4">
        <div className="max-w-4xl mx-auto space-y-3">
          {/* Suggested Prompts */}
          {messages.length === 0 && (
            <div className="flex gap-2 flex-wrap">
              {suggestedPrompts.map((prompt, idx) => (
                <Button
                  key={idx}
                  variant="outline"
                  size="sm"
                  className="bg-[#1a1a1a] border-[#2a2a2a] text-gray-300 hover:bg-[#2a2a2a] hover:text-white gap-2"
                  onClick={() => setInput(prompt.text)}
                >
                  <prompt.icon className="w-3 h-3" />
                  {prompt.text}
                </Button>
              ))}
            </div>
          )}

          {/* Input Box */}
          <div className="relative flex items-end gap-2">
            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type your message here... (e.g., write code, analyze data, generate content)"
              className="min-h-[56px] max-h-[200px] bg-[#1a1a1a] border-[#2a2a2a] text-white placeholder:text-gray-500 resize-none pr-12"
              disabled={isLoading}
            />
            <Button
              onClick={sendMessage}
              disabled={!input.trim() || isLoading}
              size="icon"
              className="bg-[#10b981] hover:bg-[#059669] text-white h-[56px] w-[56px] flex-shrink-0"
            >
              <Send className="h-5 w-5" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
