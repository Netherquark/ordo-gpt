import { useState } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Sparkles } from 'lucide-react';

interface LoginPageProps {
  onLogin: (username: string, apiKey: string) => void;
}

export function LoginPage({ onLogin }: LoginPageProps) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [error, setError] = useState('');

  const handleLogin = () => {
    if (!username.trim()) {
      setError('Username is required');
      return;
    }
    if (!password.trim()) {
      setError('Password is required');
      return;
    }
    
    // Simple demo authentication - in production, this would be a real API call
    if (password.length >= 6) {
      onLogin(username, apiKey || 'demo-api-key');
    } else {
      setError('Password must be at least 6 characters');
    }
  };

  const handleApiKeyLogin = () => {
    if (!apiKey.trim()) {
      setError('API Key is required');
      return;
    }
    onLogin('API User', apiKey);
  };

  return (
    <div className="min-h-screen bg-[#0a0a0a] flex items-center justify-center p-4">
      <div className="w-full max-w-md space-y-6">
        {/* Logo and Header */}
        <div className="text-center space-y-4">
          <div className="w-16 h-16 bg-[#10b981] rounded-2xl flex items-center justify-center mx-auto">
            <Sparkles className="w-10 h-10 text-white" />
          </div>
          <div>
            <h1 className="text-3xl text-white">HNet Platform</h1>
            <p className="text-gray-400 mt-2">AI-Powered Language Model Interface</p>
          </div>
        </div>

        {/* Login Card */}
        <Card className="bg-[#0f0f0f] border-[#1a1a1a]">
          <Tabs defaultValue="login" className="w-full">
            <CardHeader>
              <TabsList className="grid w-full grid-cols-2 bg-[#1a1a1a]">
                <TabsTrigger value="login" className="data-[state=active]:bg-[#10b981]">
                  Login
                </TabsTrigger>
                <TabsTrigger value="apikey" className="data-[state=active]:bg-[#10b981]">
                  API Key
                </TabsTrigger>
              </TabsList>
            </CardHeader>

            <TabsContent value="login">
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="username" className="text-gray-300">Username</Label>
                  <Input
                    id="username"
                    type="text"
                    placeholder="Enter your username"
                    value={username}
                    onChange={(e) => {
                      setUsername(e.target.value);
                      setError('');
                    }}
                    className="bg-[#1a1a1a] border-[#2a2a2a] text-white"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="password" className="text-gray-300">Password</Label>
                  <Input
                    id="password"
                    type="password"
                    placeholder="Enter your password"
                    value={password}
                    onChange={(e) => {
                      setPassword(e.target.value);
                      setError('');
                    }}
                    onKeyDown={(e) => e.key === 'Enter' && handleLogin()}
                    className="bg-[#1a1a1a] border-[#2a2a2a] text-white"
                  />
                </div>
                {error && <p className="text-red-500 text-sm">{error}</p>}
                <div className="pt-2">
                  <Button
                    onClick={handleLogin}
                    className="w-full bg-[#10b981] hover:bg-[#059669] text-white"
                  >
                    Sign In
                  </Button>
                </div>
              </CardContent>
              <CardFooter className="flex flex-col space-y-2">
                <p className="text-xs text-gray-500 text-center">
                  Demo: Use any username with password (min 6 chars)
                </p>
              </CardFooter>
            </TabsContent>

            <TabsContent value="apikey">
              <CardContent className="space-y-4">
                <CardDescription className="text-gray-400">
                  Connect using your HNet API key for direct access
                </CardDescription>
                <div className="space-y-2">
                  <Label htmlFor="apikey" className="text-gray-300">API Key</Label>
                  <Input
                    id="apikey"
                    type="password"
                    placeholder="hnet_xxxxxxxxxxxxx"
                    value={apiKey}
                    onChange={(e) => {
                      setApiKey(e.target.value);
                      setError('');
                    }}
                    onKeyDown={(e) => e.key === 'Enter' && handleApiKeyLogin()}
                    className="bg-[#1a1a1a] border-[#2a2a2a] text-white"
                  />
                </div>
                {error && <p className="text-red-500 text-sm">{error}</p>}
                <div className="pt-2">
                  <Button
                    onClick={handleApiKeyLogin}
                    className="w-full bg-[#10b981] hover:bg-[#059669] text-white"
                  >
                    Connect with API Key
                  </Button>
                </div>
              </CardContent>
              <CardFooter>
                <p className="text-xs text-gray-500 text-center w-full">
                  Demo: Enter any API key to continue
                </p>
              </CardFooter>
            </TabsContent>
          </Tabs>
        </Card>

        {/* Features */}
        <div className="grid grid-cols-3 gap-4 text-center">
          <div className="space-y-1">
            <div className="text-[#10b981]">Fast</div>
            <div className="text-xs text-gray-500">Low latency</div>
          </div>
          <div className="space-y-1">
            <div className="text-[#10b981]">Secure</div>
            <div className="text-xs text-gray-500">Encrypted</div>
          </div>
          <div className="space-y-1">
            <div className="text-[#10b981]">Scalable</div>
            <div className="text-xs text-gray-500">Cloud-ready</div>
          </div>
        </div>
      </div>
    </div>
  );
}
