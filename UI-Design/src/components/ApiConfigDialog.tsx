import { useState } from 'react';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from './ui/dialog';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Button } from './ui/button';

interface ApiConfigDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  apiEndpoint: string;
  onSave: (endpoint: string) => void;
}

export function ApiConfigDialog({ open, onOpenChange, apiEndpoint, onSave }: ApiConfigDialogProps) {
  const [endpoint, setEndpoint] = useState(apiEndpoint);

  const handleSave = () => {
    onSave(endpoint);
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-[#1a1a1a] border-[#2a2a2a] text-white">
        <DialogHeader>
          <DialogTitle>API Configuration</DialogTitle>
          <DialogDescription className="text-gray-400">
            Configure the HNet API server endpoint
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4 py-4">
          <div className="space-y-2">
            <Label htmlFor="endpoint">API Endpoint</Label>
            <Input
              id="endpoint"
              value={endpoint}
              onChange={(e) => setEndpoint(e.target.value)}
              placeholder="http://192.168.1.137:5000"
              className="bg-[#0f0f0f] border-[#2a2a2a] text-white"
            />
            <p className="text-xs text-gray-500">
              Current: http://192.168.1.137:5000 (HNet server IP)
            </p>
          </div>
          <div className="space-y-2 p-3 bg-[#0f0f0f] rounded-lg border border-[#2a2a2a]">
            <h4 className="text-sm text-[#10b981]">Available Endpoints:</h4>
            <ul className="text-xs text-gray-400 space-y-1">
              <li>• GET /v1/models - List available models</li>
              <li>• POST /v1/completions - Text completion</li>
              <li>• POST /v1/chat/completions - Chat completion</li>
            </ul>
          </div>
        </div>
        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            className="bg-transparent border-[#2a2a2a] text-gray-300 hover:bg-[#2a2a2a]"
          >
            Cancel
          </Button>
          <Button
            onClick={handleSave}
            className="bg-[#10b981] hover:bg-[#059669] text-white"
          >
            Save
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
