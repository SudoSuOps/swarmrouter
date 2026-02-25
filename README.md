# SwarmRouter-v1

**Intelligent routing for multi-model inference fleets**

SwarmRouter is a lightweight 3-4B parameter LLM fine-tuned specifically for routing queries to specialized models in a heterogeneous inference fleet. It outputs structured JSON routing decisions in <100ms, enabling cost-efficient and latency-optimized model orchestration.

## Why SwarmRouter?

Running multiple specialized models (medical 14B, research 32B, general 3B) requires intelligent routing to:
- **Minimize cost** - Route simple queries to small models, complex ones to large models
- **Optimize latency** - Match query urgency to model speed
- **Ensure safety** - Route high-risk medical/legal queries to specialized models
- **Maximize throughput** - Balance load across heterogeneous hardware

Traditional if-else routing breaks at scale. SwarmRouter learns routing physics from 60K synthetic examples and makes decisions in one forward pass.

## Architecture

```
User Query → SwarmRouter-3B → JSON Decision → Target Model
                 ↓
        {
          "domain": "medical",
          "recommended_model": "med-14b",
          "risk_level": "high",
          "reasoning": "Clinical diagnosis requires specialist model"
        }
```

**Base Model**: Qwen2.5-3B-Instruct
**Training**: QLoRA (rank 32), 60K synthetic routing examples
**Output Format**: Strict JSON with 10 required fields
**Latency**: <100ms on RTX 3090
**Hardware**: Single RTX PRO 6000 (96GB) or 2x RTX 3090 (48GB)

## Quick Start

### 1. Installation

```bash
git clone <repo>
cd swarmrouter
pip install -r requirements.txt
```

### 2. Generate Dataset

```bash
cd data
python make_dataset.py
```

This generates:
- `swarmrouter_train_60k.jsonl` - 60,000 training examples
- `swarmrouter_eval_400.jsonl` - 400 evaluation examples

### 3. Train Model

```bash
python scripts/train.py
```

Training time: ~6-8 hours on RTX PRO 6000 (96GB) or ~10-12 hours on 2x RTX 3090

### 4. Evaluate

```bash
python scripts/eval.py
```

Expected metrics:
- JSON validity: >98%
- Routing accuracy: >90%
- Model selection: >85%
- Domain classification: >95%

### 5. Run CLI

```bash
python scripts/cli_router.py
```

Interactive mode:
```
Query> What are the differential diagnoses for chest pain?

🧭 Routing Decision
======================================================================
Domain:           medical
Task Type:        reasoning
Complexity:       high
Risk Level:       high
Recommended:      med-14b
Escalation:       Yes
Tools Required:   None
Reasoning:        Clinical differential diagnosis requires medical specialist model
----------------------------------------------------------------------
Latency:          87.3ms
======================================================================
```

### 6. Serve API

```bash
python scripts/serve_api.py --host 0.0.0.0 --port 8000
```

API usage:
```bash
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain GPU memory optimization"}'
```

Response:
```json
{
  "routing": {
    "domain": "compute",
    "task_type": "qa",
    "complexity": "medium",
    "risk_level": "low",
    "latency_tier": "standard",
    "cost_sensitivity": "medium",
    "recommended_model": "research-8b",
    "escalation_allowed": true,
    "requires_tools": [],
    "reasoning": "Technical compute question suitable for research model"
  },
  "latency_ms": 92.1,
  "model_version": "swarmrouter-v1"
}
```

## Routing Schema

Every routing decision contains 10 required fields:

```python
{
  "domain": str,              # general|coding|operations|research|medical|cre|aviation|compute
  "task_type": str,           # qa|summarization|reasoning|generation|planning|triage
  "complexity": str,          # low|medium|high
  "risk_level": str,          # low|medium|high
  "latency_tier": str,        # realtime|standard|batch
  "cost_sensitivity": str,    # low|medium|high
  "recommended_model": str,   # router-3b|research-8b|med-14b|research-32b
  "escalation_allowed": bool, # Can escalate to larger model if needed
  "requires_tools": list,     # ["calculator", "search", "code_interpreter"]
  "reasoning": str            # Brief explanation (≤120 chars)
}
```

## Routing Physics

The model learns these routing rules from 60K examples:

### Model Selection

| Domain | Complexity | Risk | Model |
|--------|-----------|------|-------|
| Medical | Any | High | `med-14b` |
| Medical | Any | Medium/Low | `research-8b` |
| Aviation | High | High | `research-32b` |
| Aviation | Medium/Low | Any | `research-8b` |
| CRE | High | Any | `research-32b` |
| Research | High | Any | `research-32b` |
| Compute | High | Any | `research-8b` |
| Coding | High | Any | `research-8b` |
| Operations | Any | Any | `research-8b` |
| General | Low | Low | `router-3b` |

### Domain Distribution

Training data is balanced across domains:
- Medical: 20%
- Research: 20%
- Compute: 15%
- CRE: 15%
- Aviation: 10%
- Coding: 10%
- Operations: 5%
- General: 5%

## Advanced Usage

### Merge LoRA Adapters

For faster inference, merge LoRA into base model:

```bash
python scripts/merge_lora.py --quantization both
```

Outputs:
- `models/swarmrouter-v1-merged/bf16/` - BF16 merged model
- `models/swarmrouter-v1-merged/4bit/` - 4-bit quantized model

### Export to GGUF

For llama.cpp or Ollama deployment:

```bash
# First merge LoRA
python scripts/merge_lora.py

# Then export to GGUF
python scripts/export_gguf.py \
  --model-path ./models/swarmrouter-v1-merged/bf16 \
  --quantization Q4_K_M
```

Use with Ollama:
```bash
# Create Modelfile
cat > Modelfile << EOF
FROM ./models/swarmrouter-v1_Q4_K_M.gguf
EOF

# Import
ollama create swarmrouter -f Modelfile

# Run
ollama run swarmrouter
```

### Fuzzing & Adversarial Testing

Test robustness against edge cases:

```bash
python scripts/router_fuzzer.py
```

Tests include:
- Empty/malformed inputs
- Prompt injection attempts
- Multilingual queries
- Ambiguous requests
- Mixed-domain questions
- Jailbreak attempts

Expected pass rate: >85%

## Configuration

Edit `configs/train.yaml` to customize training:

```yaml
model:
  base_model: "Qwen/Qwen2.5-3B-Instruct"
  load_in_4bit: true

lora:
  r: 32
  lora_alpha: 32
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 8  # Effective batch size = 32
  learning_rate: 2.0e-4
```

## Project Structure

```
swarmrouter/
├── configs/
│   └── train.yaml              # Training configuration
├── data/
│   ├── topics/                 # Topic lists for each domain
│   │   ├── medical.txt
│   │   ├── aviation.txt
│   │   └── ...
│   ├── templates.json          # Prompt templates
│   ├── generator.py            # Synthetic data generator
│   └── make_dataset.py         # Dataset creation script
├── scripts/
│   ├── train.py                # Training script
│   ├── eval.py                 # Evaluation script
│   ├── router_fuzzer.py        # Adversarial testing
│   ├── serve_api.py            # FastAPI server
│   ├── cli_router.py           # CLI tool
│   ├── merge_lora.py           # Merge LoRA adapters
│   └── export_gguf.py          # GGUF export
├── swarmrouter/
│   ├── __init__.py
│   ├── schema.py               # Pydantic models
│   └── routing_rules.py        # Routing physics
├── models/                     # Trained models (created during training)
├── outputs/                    # Evaluation results (created during eval)
├── requirements.txt
└── README.md
```

## Hardware Requirements

### Training
- **Minimum**: 1x RTX 3090 (24GB VRAM)
- **Recommended**: 1x RTX PRO 6000 (96GB VRAM)
- **RAM**: 32GB system memory
- **Storage**: 50GB free space

### Inference
- **Minimum**: 1x RTX 3060 (12GB VRAM) for 4-bit
- **Recommended**: 1x RTX 3090 (24GB VRAM) for BF16
- **Latency**: 60-100ms on RTX 3090

## Performance Benchmarks

| Hardware | Format | Latency | Throughput |
|----------|--------|---------|------------|
| RTX PRO 6000 | BF16 | 45ms | 22 queries/sec |
| RTX 3090 | BF16 | 87ms | 11 queries/sec |
| RTX 3090 | 4-bit | 62ms | 16 queries/sec |
| RTX 3060 Ti | 4-bit | 105ms | 9 queries/sec |

## Dataset Statistics

The synthetic dataset is generated with deterministic seeding for reproducibility:

**Training Set (60K samples)**:
- Medical: 12,000 (20%)
- Research: 12,000 (20%)
- Compute: 9,000 (15%)
- CRE: 9,000 (15%)
- Aviation: 6,000 (10%)
- Coding: 6,000 (10%)
- Operations: 3,000 (5%)
- General: 3,000 (5%)

**Complexity Distribution**:
- Low: 12,000 (20%)
- Medium: 30,000 (50%)
- High: 18,000 (30%)

**Evaluation Set (400 samples)**:
- Balanced across all domains
- Edge cases included
- Different random seed from training

## Integration Examples

### Python Client

```python
import requests

def route_query(query: str) -> dict:
    response = requests.post(
        "http://localhost:8000/route",
        json={"message": query}
    )
    return response.json()

# Example
routing = route_query("How do I optimize GPU memory for training?")
print(f"Route to: {routing['routing']['recommended_model']}")
print(f"Latency: {routing['latency_ms']:.1f}ms")
```

### Load Balancer Integration

```python
from swarmrouter import RouterCLI

# Initialize router
router = RouterCLI(model_path="./models/swarmrouter-v1/final")
router.load_model()

# Route query
routing, latency = router.generate_routing(user_query)

# Select backend
model_endpoints = {
    "router-3b": "http://localhost:8001",
    "research-8b": "http://localhost:8002",
    "med-14b": "http://localhost:8003",
    "research-32b": "http://localhost:8004"
}

target_url = model_endpoints[routing["recommended_model"]]
# Forward to target model...
```

## Troubleshooting

### OOM During Training
- Reduce `per_device_train_batch_size` in `configs/train.yaml`
- Enable gradient checkpointing (already enabled by default)
- Use 4-bit quantization (already enabled by default)

### Low Routing Accuracy
- Increase training epochs (default: 3)
- Increase LoRA rank (default: 32)
- Generate more training data (default: 60K)

### Slow Inference
- Use merged model instead of LoRA adapters
- Use 4-bit quantization
- Batch multiple queries together

### JSON Parsing Failures
- Model may need more training
- Try temperature=0.1 for more deterministic output
- Run fuzzing tests to identify patterns

## Production Deployment

### Docker

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "scripts/serve_api.py", "--host", "0.0.0.0", "--port", "8000"]
```

### Systemd Service

```ini
[Unit]
Description=SwarmRouter API
After=network.target

[Service]
Type=simple
User=swarm
WorkingDirectory=/opt/swarmrouter
Environment="MODEL_PATH=/opt/swarmrouter/models/swarmrouter-v1/final"
ExecStart=/opt/swarmrouter/venv/bin/python scripts/serve_api.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

### Monitoring

Key metrics to track:
- Latency (p50, p95, p99)
- JSON validity rate
- Model selection distribution
- Error rate by domain
- GPU memory utilization

## License

Apache 2.0

## Citation

```bibtex
@software{swarmrouter2024,
  title={SwarmRouter: Intelligent Routing for Multi-Model Inference},
  author={Swarm & Bee},
  year={2024},
  url={https://github.com/swarmandbee/swarmrouter}
}
```

## Support

- Issues: GitHub Issues
- Documentation: This README
- API Reference: `/docs` endpoint when running serve_api.py

## Roadmap

- [ ] Multi-GPU inference support
- [ ] Batch processing mode
- [ ] Real-time performance analytics
- [ ] Auto-scaling based on load
- [ ] Integration with Ray Serve
- [ ] A/B testing framework
- [ ] Model distillation to 1B

---

Built by [Swarm & Bee](https://swarmandbee.com) - Last mile intelligence.
