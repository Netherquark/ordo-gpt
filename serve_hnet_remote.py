# serve_hnet_remote.py
# Minimal OpenAI-like server for HNet checkpoint.
# Works in hnet-convert env. Provides /v1/models, /v1/completions, /v1/chat/completions
# and robust byte-level decoding for H-Net outputs.

import json
import torch
import argparse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
from typing import Optional, List, Any
from contextlib import nullcontext

# Import tokenizer & model classes from your repo
from hnet.utils.tokenizers import ByteTokenizer
from hnet.models.config_hnet import AttnConfig, SSMConfig, HNetConfig
from hnet.models.mixer_seq import HNetForCausalLM

# Optional bitsandbytes runtime quant support
try:
    from bitsandbytes import nn as bnb_nn  # type: ignore
    HAS_BNB = True
except Exception:
    bnb_nn = None
    HAS_BNB = False

# Autocast detection (support multiple torch versions)
try:
    from torch import autocast as _autocast
    def make_autocast(device_type="cuda", dtype=torch.float16):
        return _autocast(device_type=device_type, dtype=dtype)
except Exception:
    try:
        from torch.cuda.amp import autocast as _autocast
        def make_autocast(device_type="cuda", dtype=torch.float16):
            return _autocast(device_type=device_type, dtype=dtype)
    except Exception:
        def make_autocast(device_type="cuda", dtype=torch.float16):
            return nullcontext()

app = FastAPI(title="HNet OpenAI-compatible Adapter")
MODEL = None
TOKENIZER = None
DEVICE = None
MODEL_ID = "hnet-neox"

class SubnetCORSMiddleware(CORSMiddleware):
    def is_origin_allowed(self, origin: str) -> bool:
        if origin.startswith("http://localhost") or origin.startswith("http://127.0.0.1"):
            return True
        # Allow anything from 192.168.1.*
        match = re.match(r"^https?://192\.168\.1\.\d{1,3}(:\d+)?$", origin)
        return bool(match)

app.add_middleware(
    SubnetCORSMiddleware,
    allow_origins=["*"],  # Not used directly; overridden by is_origin_allowed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0

def _force_model_half_and_buffers(model: torch.nn.Module):
    """Ensure parameters and buffers on CUDA are float16 (best-effort)."""
    try:
        model.half()
    except Exception as e:
        print("Warning: model.half() failed:", e)
    for name, buf in model.named_buffers(recurse=True):
        if isinstance(buf, torch.Tensor) and buf.dtype == torch.float32:
            try:
                buf.data = buf.data.half()
            except Exception:
                pass

def build_model(model_path: str, config_path: str, device: str = "cuda", dtype=torch.float16, quantize: bool = False):
    with open(config_path, "r") as f:
        cfg = json.load(f)
    attn_cfg = AttnConfig(**cfg.pop("attn_cfg"))
    ssm_cfg = SSMConfig(**cfg.pop("ssm_cfg"))
    hnet_cfg = HNetConfig(**cfg, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)

    print("Constructing model (device=%s, dtype=%s) ..." % (device, dtype))
    model = HNetForCausalLM(hnet_cfg, device="cpu", dtype=torch.float32)
    print("Model object constructed (CPU).")

    if quantize:
        if not HAS_BNB:
            print("Warning: quantize requested but bitsandbytes not available. Continuing without quantization.")
        else:
            print("Replacing Linear -> Linear8bitLt (bitsandbytes) at runtime (may increase load time).")
            replaced = 0
            from torch import nn
            def replace_mod(m):
                nonlocal replaced
                for name, child in list(m.named_children()):
                    replace_mod(child)
                    if isinstance(child, nn.Linear):
                        try:
                            if hasattr(bnb_nn.Linear8bitLt, "from_float"):
                                new_mod = bnb_nn.Linear8bitLt.from_float(child)
                            else:
                                new_mod = bnb_nn.Linear8bitLt(child.in_features, child.out_features, bias=(child.bias is not None))
                                with torch.no_grad():
                                    new_mod.weight.data = child.weight.data.to(new_mod.weight.data.dtype)
                                    if child.bias is not None:
                                        new_mod.bias.data = child.bias.data.to(new_mod.bias.data.dtype)
                            setattr(m, name, new_mod)
                            replaced += 1
                        except Exception as e:
                            print("Warning: replacement failed for", name, e)
            replace_mod(model)
            print(f"Replaced approx {replaced} Linear modules.")

    print("Loading state_dict from:", model_path)
    state = torch.load(model_path, map_location="cpu")
    try:
        model.load_state_dict(state)
        print("Loaded state dict (strict).")
    except Exception as e:
        print("Strict load failed:", e)
        res = model.load_state_dict(state, strict=False)
        print("Load result:", res)

    if device == "cuda" and torch.cuda.is_available():
        device_obj = torch.device("cuda")
        model.to(device_obj)
        _force_model_half_and_buffers(model)
        print("Moved model to CUDA and converted to float16.")
    else:
        print("Running on CPU (device=%s)." % device)

    model.eval()
    return model

def _safe_decode_bytes_from_tokens(tokenizer: ByteTokenizer, tokens: List[int]) -> str:
    """
    Convert generated token list -> bytes, strip BOS/EOS markers, then decode to str with robust fallbacks.
    Returns a text string suitable for JSON responses.
    """
    # Remove leading BOS/EOS tokens if present and stop at EOS token
    bos = getattr(tokenizer, "bos_idx", None)
    eos = getattr(tokenizer, "eos_idx", None)

    cleaned = []
    for t in tokens:
        if t == bos:
            continue
        if t == eos:
            break
        # ensure ints in 0-255
        cleaned.append(int(t) & 0xFF)

    bseq = bytes(cleaned)

    # Try UTF-8 first (preferred), fall back to replacement, then latin-1 (lossless 1:1 byte->unicode)
    try:
        return bseq.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return bseq.decode("utf-8", errors="replace")
        except Exception:
            return bseq.decode("latin-1", errors="replace")

@app.get("/v1/models")
async def list_models():
    # Minimal model metadata so clients (OpenWebUI) can list available models.
    return {"data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}]}

@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    global MODEL, TOKENIZER, DEVICE
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    prompt = req.prompt
    max_tokens = int(req.max_tokens or 128)
    temperature = float(req.temperature or 1.0)
    top_p = float(req.top_p or 1.0)

    tokenizer = TOKENIZER
    device = DEVICE

    encoded = tokenizer.encode([prompt], add_bos=True)[0]
    input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long, device=device).unsqueeze(0)

    cache_dtype = torch.float16 if device.type == "cuda" else torch.float32
    inference_cache = MODEL.allocate_inference_cache(1, input_ids.shape[1] + max_tokens, dtype=cache_dtype)

    autocast_cm = make_autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()

    with torch.inference_mode():
        mask = torch.ones(input_ids.shape, device=device, dtype=torch.bool)
        with autocast_cm:
            output = MODEL.forward(input_ids, mask=mask, inference_params=inference_cache)

    logits = output.logits[0, -1, :] / max(temperature, 1e-8)
    generated = []
    for _ in range(max_tokens):
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        if next_token.item() == tokenizer.eos_idx:
            break
        generated.append(next_token.item())

        current_token = next_token.unsqueeze(0).to(device)
        with torch.inference_mode():
            with autocast_cm:
                out = MODEL.step(current_token, inference_cache)
        logits = out.logits[0, -1, :] / max(temperature, 1e-8)

    text = _safe_decode_bytes_from_tokens(tokenizer, generated)
    return {"id": MODEL_ID, "object": "text_completion", "choices": [{"text": text, "index": 0}], "usage": {"prompt_tokens": len(encoded["input_ids"]), "completion_tokens": len(generated)}}

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    """
    Convert OpenAI-style chat request -> single prompt completion by concatenating messages.
    Responds with OpenAI-like chat response object.
    """
    # Basic concatenation strategy: role/content pairs -> single prompt.
    prompt_parts = []
    for m in req.messages:
        # keep simple format like "role: content\n"
        role = m.role or "user"
        prompt_parts.append(f"{role}: {m.content}")
    prompt = "\n".join(prompt_parts) + "\nassistant:"

    comp_req = CompletionRequest(prompt=prompt, max_tokens=req.max_tokens, temperature=req.temperature, top_p=req.top_p)
    res = await completions(comp_req)

    content = ""
    # completions returns "choices"[0]["text"]
    if isinstance(res, dict) and res.get("choices"):
        content = res["choices"][0].get("text", "")

    # Build OpenAI chat-like response
    return {
        "id": MODEL_ID,
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": res.get("usage", {}) if isinstance(res, dict) else {}
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=os.environ.get("HNET_MODEL_PATH", "/home/netherquark/models/hnet_1stage_L.pt"))
    parser.add_argument("--config-path", type=str, default="hnet/configs/hnet_1stage_L.json")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--quantize", action="store_true", help="Replace linear modules at runtime with bitsandbytes wrappers")
    args = parser.parse_args()

    global MODEL, TOKENIZER, DEVICE
    TOKENIZER = ByteTokenizer()
    DEVICE = torch.device(args.device if (args.device=="cpu" or torch.cuda.is_available()) else "cpu")

    MODEL = build_model(args.model_path, args.config_path, device=("cuda" if DEVICE.type=="cuda" else "cpu"), dtype=torch.float16, quantize=args.quantize)
    print("Model loaded and ready. Listening on %s:%d" % (args.host, args.port))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()