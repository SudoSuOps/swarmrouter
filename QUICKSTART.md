# SwarmRouter Quick Start Guide

This guide will get you from zero to trained router in 30 minutes.

## Prerequisites

- Ubuntu 20.04+ or similar Linux
- Python 3.11+
- CUDA 12.1+ (for GPU training)
- 1x RTX 3090 (24GB) minimum or RTX PRO 6000 (96GB) recommended
- 50GB free disk space

## Step 1: Install Dependencies (1 minute)

```bash
cd /home/swarm/Desktop/swarmrouter

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync lightweight deps (API, CLI, data tools)
uv sync

# For GPU training rigs (swarmrails, whale):
uv sync --group train

# Verify setup
uv run python3 test_setup.py
```

**Expected output**: "✅ All tests passed! SwarmRouter is ready."

## Step 2: Generate Dataset (5 minutes)

```bash
cd data
python3 make_dataset.py
```

**Output**:
- `swarmrouter_train_60k.jsonl` (~45MB)
- `swarmrouter_eval_400.jsonl` (~300KB)

**What's happening**: Generator creates 60K synthetic routing examples with balanced domain distribution and deterministic seeding.

## Step 3: Train Model (6-12 hours)

```bash
# Start training
python3 scripts/train.py

# Monitor progress (in another terminal)
watch -n 5 'tail -20 models/swarmrouter-v1/final/trainer_log.jsonl'
```

**Training time**:
- RTX PRO 6000 (96GB): ~6 hours
- RTX 3090 (24GB): ~10 hours
- 2x RTX 3090: ~8 hours

**What's happening**: QLoRA fine-tuning of Qwen2.5-3B-Instruct on 60K routing examples for 3 epochs.

**Output**: `models/swarmrouter-v1/final/`

## Step 4: Evaluate (5 minutes)

```bash
python3 scripts/eval.py
```

**Expected metrics**:
- JSON validity: >98%
- Full accuracy: >90%
- Model routing: >85%
- Domain classification: >95%

**Output**: `outputs/eval_results.json`

## Step 5: Test Interactive CLI (2 minutes)

```bash
python3 scripts/cli_router.py
```

**Try these queries**:
```
Query> What are the symptoms of pneumonia?
# Should route to: med-14b

Query> How do I optimize GPU memory?
# Should route to: research-8b

Query> What is 2+2?
# Should route to: router-3b

Query> Analyze this commercial real estate deal: $5M property, 8% cap rate
# Should route to: research-32b

Query> /quit
```

## Step 6: Run API Server (1 minute)

```bash
python3 scripts/serve_api.py --host 0.0.0.0 --port 8000
```

**Test in another terminal**:
```bash
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain model parallelism"}'
```

**Expected response**:
```json
{
  "routing": {
    "domain": "compute",
    "recommended_model": "research-8b",
    ...
  },
  "latency_ms": 87.3
}
```

## Optional: Advanced Steps

### Run Adversarial Tests

```bash
python3 scripts/router_fuzzer.py
```

Tests edge cases, injections, malformed inputs. Expected pass rate: >85%

### Merge LoRA for Faster Inference

```bash
python3 scripts/merge_lora.py --quantization both
```

Creates merged models in `models/swarmrouter-v1-merged/`

### Export to GGUF for llama.cpp

```bash
# Install llama.cpp first
git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp
cd ~/llama.cpp && make
pip install -r requirements.txt

# Export
cd /home/swarm/Desktop/swarmrouter
python3 scripts/export_gguf.py \
  --model-path ./models/swarmrouter-v1-merged/bf16 \
  --quantization Q4_K_M
```

Use with Ollama:
```bash
# Create Modelfile
echo "FROM ./models/swarmrouter-v1_Q4_K_M.gguf" > Modelfile

# Import
ollama create swarmrouter -f Modelfile

# Run
ollama run swarmrouter "What are the symptoms of COVID-19?"
```

## Troubleshooting

### "CUDA out of memory"
- Reduce batch size in `configs/train.yaml`: `per_device_train_batch_size: 2`
- Reduce `gradient_accumulation_steps: 4`
- Use smaller model or more GPUs

### "Module not found" errors
```bash
uv sync          # lightweight
uv sync --group train  # ML training deps
```

### Training stuck or slow
- Check GPU utilization: `watch -n 1 nvidia-smi`
- Should see >95% GPU util during training
- If low, check if other processes are using GPU

### Low eval accuracy (<80%)
- Train for more epochs: Edit `configs/train.yaml`, set `num_train_epochs: 5`
- Increase LoRA rank: Set `r: 64` in config
- Generate more data: `python3 data/make_dataset.py --train-size 100000`

## Production Checklist

Before deploying to production:

- [ ] Evaluation accuracy >90%
- [ ] Fuzzing pass rate >85%
- [ ] Latency <100ms on target hardware
- [ ] API health endpoint returns 200
- [ ] Systemd service configured
- [ ] Monitoring/alerting set up
- [ ] Load testing completed
- [ ] Backup/restore tested

## Next Steps

1. **Integrate with your fleet**: Connect router to your model endpoints
2. **Monitor performance**: Track latency, accuracy, cost savings
3. **Iterate**: Fine-tune routing rules based on production data
4. **Scale**: Deploy multiple router instances behind load balancer

## Support

- **Docs**: See README.md for full documentation
- **Issues**: Open GitHub issue
- **Examples**: Check `scripts/` for integration examples

---

Built by Swarm & Bee - Last mile intelligence.
