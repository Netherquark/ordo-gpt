# Ordo-GPT: Hierarchical Chunking Framework for GPT-NeoX

## Ordo-GPT

### Overview

**Ordo-GPT** is a modular deep learning project focused on enhancing text generation by **applying hierarchical chunking to GPT-NeoX**.
Built around the **HNet (Hierarchical Network)** architecture, it introduces a multi-level context processing mechanism that enables structured and context-aware text generation even across long sequences.

This framework provides flexible components, utilities, and configuration tools for developing, training, evaluating, and serving transformer-like models optimized for natural language understanding and sequence modeling.

---

### Features

* Implements **Hierarchical Chunking** with GPT-NeoX style attention.
* Modular deep learning framework with clearly defined architecture.
* Configurable **HNet** architecture with JSON-based model setup.
* Utilities for **training**, **tokenization**, and **model evaluation**.
* **Text generation engine** with prompt-based inference (`generate.py`).
* **Remote model serving** through an API interface (`serve_hnet_remote.py`).
* Extensible architecture for easy experimentation with new layers or configurations.

---

### Installation (Regular)

1. Clone the repository:

   ```bash
   git clone https://github.com/netherquark/ordo-gpt.git
   ```

2. Navigate to the project directory:

   ```bash
   cd ordo-gpt/src/hnet
   ```

3. Install the project in editable mode:

   ```bash
   pip install -e .
   ```

---

### Usage (Regular)

#### Running the API Server

Launch the model inference server:

```bash
python serve_hnet_remote.py --model-path ~/models/hnet_neox.pt --config-path configs/hnet_neox.json --device cuda --quantize
```

Send a request using:

```bash
curl -s -X POST "http://localhost:5000/v1/completions"   -H "Content-Type: application/json"   -d '{
    "prompt": "Glass is an amorphous (non-crystalline) solid. Because it is often transparent and chemically inert, glass has found widespread practical, technological, and decorative use in window panes, tableware, and optics. Some common objects made of glass are ",
    "max_tokens": 256,
    "temperature": 0.35
  }' | jq
```

---

### Configuration Example

```json
{
    "arch_layout": ["m4", ["T1m4", ["T26"], "m4T1"], "m4"],
    "d_model": [1024, 1024, 1536],
    "d_intermediate": [0, 2816, 4096],
    "vocab_size": 256,
    "ssm_cfg": {
        "chunk_size": 256,
        "d_conv": 4,
        "d_state": 128,
        "expand": 2
    },
    "attn_cfg": {
        "num_heads": [16, 16, 16],
        "rotary_emb_dim": [32, 32, 48],
        "window_size": [1023, 1023, -1]
    },
    "tie_embeddings": false
}
```

---

### License

This project is licensed under the **GPLv3 License**.
Refer to the `LICENSE` file for more details.

