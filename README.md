# Ordo-GPT: Hierarchical Chunking Framework for GPT-NeoX

## Ordo-GPT

### Overview

**Ordo-GPT** is a modular deep learning project focused on enhancing text generation by **applying hierarchical chunking to GPT-NeoX**.
Built around the **HNet (Hierarchical Network)** architecture, it introduces a multi-level context processing mechanism that enables structured and context-aware text generation even across long sequences.

This framework provides flexible components, utilities, and configuration tools for developing, training, evaluating, and serving transformer-like models optimized for natural language understanding and sequence modeling.

---

### Features

* Implements **Hierarchical Chunking** to extend GPT-NeoX’s contextual understanding.
* Modular deep learning framework with clearly defined architecture.
* Configurable **HNet** architecture with JSON-based model setup.
* Utilities for **training**, **tokenization**, and **model evaluation**.
* **Text generation engine** with prompt-based inference (`generate.py`).
* **Remote model serving** through an API interface (`serve_hnet_remote.py`).
* Support for **large-scale model configurations** (e.g., `hnet_1stage_XL`).
* Extensible architecture for easy experimentation with new layers or configurations.

---

### Installation (Regular)

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/ordo-gpt.git
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

#### Training the Model

Run the training pipeline using:

```bash
python src/hnet/hnet/utils/train.py --config src/hnet/configs/hnet_1stage_XL.json
```

You can modify hyperparameters in the configuration file such as:

* Number of layers
* Hidden dimension size
* Maximum sequence length
* Dropout and learning rate

#### Generating Text

Generate text using a pretrained model with:

```bash
python src/hnet/generate.py --prompt "Artificial intelligence is transforming"
```

**Example Output:**

```
Artificial intelligence is transforming every domain by introducing adaptive systems capable of reasoning through structured context.
```

#### Running the API Server

Launch the model inference server:

```bash
python serve_hnet_remote.py --host 0.0.0.0 --port 8000
```

Send a request using:

```bash
curl -X POST http://localhost:8000/generate -d '{"prompt": "Hello world"}'
```

---

### Evaluating Models

Evaluate performance, analyze loss, and validate outputs by extending or customizing the training script under:

```
src/hnet/hnet/utils/train.py
```

You can also visualize token-level coherence and generation patterns using the hierarchical chunking structure.

---

### Configuration Example

```json
{
  "model_name": "hnet_1stage_XL",
  "hidden_dim": 2048,
  "num_layers": 24,
  "vocab_size": 50257,
  "max_seq_len": 1024,
  "dropout": 0.1
}
```

---

### Dependencies

* Python 3.11
* PyTorch
* Transformers
* Tokenizers
* NumPy
* tqdm
* Flask

---

### License

This project is licensed under the **MIT License**.
Refer to the `LICENSE` file for more details.

