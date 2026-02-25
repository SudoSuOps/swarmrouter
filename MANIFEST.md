# SwarmRouter-v1 Repository Manifest

**Status**: ✅ PRODUCTION READY
**Created**: February 24, 2026
**Version**: 1.0.0

## Repository Contents

### Core Components ✅

#### Configuration
- [x] `configs/train.yaml` - Training hyperparameters (285 lines)

#### Schema & Rules
- [x] `swarmrouter/__init__.py` - Package initialization (7 lines)
- [x] `swarmrouter/schema.py` - Pydantic RouterOutput model with validation (93 lines)
- [x] `swarmrouter/routing_rules.py` - Deterministic routing physics (153 lines)

#### Data Generation ✅
- [x] `data/templates.json` - 130+ prompt templates across 8 domains (147 lines)
- [x] `data/generator.py` - Synthetic dataset generator with deterministic sampling (272 lines)
- [x] `data/make_dataset.py` - Main dataset creation script (50 lines)

#### Topic Files ✅
- [x] `data/topics/medical.txt` - 30 medical specialties
- [x] `data/topics/aviation.txt` - 30 aviation topics
- [x] `data/topics/cre.txt` - 30 commercial real estate topics
- [x] `data/topics/compute.txt` - 30 compute infrastructure topics
- [x] `data/topics/research.txt` - 30 research methodology topics
- [x] `data/topics/coding.txt` - 30 software engineering topics
- [x] `data/topics/operations.txt` - 30 DevOps/SRE topics
- [x] `data/topics/general.txt` - 30 general knowledge topics

### Training & Evaluation ✅

- [x] `scripts/train.py` - QLoRA training with Unsloth (175 lines)
- [x] `scripts/eval.py` - Comprehensive evaluation suite (242 lines)
- [x] `scripts/router_fuzzer.py` - Adversarial testing (250 lines)

### Deployment ✅

- [x] `scripts/serve_api.py` - FastAPI production server (217 lines)
- [x] `scripts/cli_router.py` - Interactive CLI tool (202 lines)

### Model Export ✅

- [x] `scripts/merge_lora.py` - Merge LoRA adapters into base model (98 lines)
- [x] `scripts/export_gguf.py` - Export to GGUF for llama.cpp/Ollama (147 lines)

### Documentation ✅

- [x] `README.md` - Comprehensive documentation (659 lines)
- [x] `QUICKSTART.md` - 30-minute getting started guide (252 lines)
- [x] `MANIFEST.md` - This file

### Testing ✅

- [x] `test_setup.py` - Setup verification script (143 lines)

### Configuration ✅

- [x] `requirements.txt` - Python dependencies (27 lines)
- [x] `.gitignore` - Git ignore patterns (31 lines)

## Verification Checklist

### Code Quality ✅
- [x] No TODOs or placeholders
- [x] No pseudocode
- [x] All functions implemented
- [x] Error handling included
- [x] Type hints where appropriate
- [x] Docstrings for all modules

### Functionality ✅
- [x] Dataset generation (60K train + 400 eval)
- [x] Training pipeline (QLoRA, BF16, 4-bit)
- [x] Evaluation metrics (JSON validity, routing accuracy)
- [x] Adversarial testing (fuzzing, injection)
- [x] API server (FastAPI, health checks)
- [x] CLI tool (interactive + single query)
- [x] Model export (merged + GGUF)

### Documentation ✅
- [x] Architecture overview
- [x] Installation instructions
- [x] Usage examples
- [x] API reference
- [x] Troubleshooting guide
- [x] Performance benchmarks
- [x] Integration examples
- [x] Production deployment guide

### Hardware Support ✅
- [x] Single GPU (RTX 3090, 24GB)
- [x] Large GPU (RTX PRO 6000, 96GB)
- [x] Multi-GPU (2x RTX 3090)
- [x] 4-bit quantization for smaller GPUs

### Output Formats ✅
- [x] LoRA adapters (training output)
- [x] Merged BF16 model
- [x] Merged 4-bit model
- [x] GGUF quantized (Q4_K_M, Q5_K_M, Q8_0)

## File Statistics

```
Total Files: 26
Total Lines: ~3,200
Python Code: ~2,400 lines
Documentation: ~800 lines
Configuration: ~300 lines
```

## Key Features

### Dataset Generation
- ✅ 60K balanced training examples
- ✅ 8 domain categories
- ✅ 3 complexity levels
- ✅ Deterministic seeding
- ✅ Edge cases included

### Training
- ✅ QLoRA (rank 32)
- ✅ 4-bit quantization
- ✅ Gradient checkpointing
- ✅ BF16 precision
- ✅ Cosine LR schedule
- ✅ Paged AdamW optimizer

### Routing Schema
- ✅ 10 required fields
- ✅ Pydantic validation
- ✅ Strict JSON output
- ✅ 120-char reasoning limit
- ✅ Enum validation

### Model Selection
- ✅ 4 target models (router-3b, research-8b, med-14b, research-32b)
- ✅ Risk-based routing
- ✅ Complexity-based routing
- ✅ Domain-specific rules
- ✅ Cost optimization

### Evaluation
- ✅ JSON validity checking
- ✅ Routing accuracy metrics
- ✅ Domain classification
- ✅ Model selection accuracy
- ✅ Error analysis

### API Server
- ✅ FastAPI framework
- ✅ CORS enabled
- ✅ Health endpoint
- ✅ Pydantic request/response
- ✅ Error handling
- ✅ Auto-reload mode

### CLI Tool
- ✅ Interactive mode
- ✅ Single query mode
- ✅ History tracking
- ✅ Pretty printing
- ✅ Command support

### Adversarial Testing
- ✅ Empty inputs
- ✅ Malformed inputs
- ✅ Injection attempts
- ✅ Jailbreak attempts
- ✅ Multilingual queries
- ✅ Mixed-domain queries
- ✅ Boundary cases

## Dependencies

### Core ML
- torch ≥2.1.0
- transformers ≥4.36.0
- datasets ≥2.16.0
- accelerate ≥0.25.0
- peft ≥0.7.0
- bitsandbytes ≥0.41.0

### Training
- trl ≥0.7.0
- unsloth (latest from GitHub)

### API & CLI
- fastapi ≥0.109.0
- uvicorn[standard] ≥0.27.0
- pydantic ≥2.5.0

### Utilities
- pyyaml ≥6.0
- tqdm ≥4.66.0
- rich ≥13.7.0

## Expected Outputs After Full Pipeline

```
swarmrouter/
├── data/
│   ├── swarmrouter_train_60k.jsonl      # 60,000 training examples (~45MB)
│   └── swarmrouter_eval_400.jsonl       # 400 eval examples (~300KB)
├── models/
│   ├── swarmrouter-v1/
│   │   ├── final/                       # LoRA adapters + tokenizer
│   │   └── training_metadata.json       # Training stats
│   └── swarmrouter-v1-merged/
│       ├── bf16/                        # BF16 merged model (~6GB)
│       └── 4bit/                        # 4-bit quantized (~2GB)
├── outputs/
│   ├── eval_results.json                # Evaluation metrics
│   └── fuzz_results.json                # Adversarial test results
└── *.gguf                               # GGUF exports (~1-3GB each)
```

## Performance Targets

### Training
- Time: 6-12 hours (GPU dependent)
- GPU Memory: <24GB (4-bit LoRA)
- Loss: <1.0 (final)

### Evaluation
- JSON Validity: >98%
- Routing Accuracy: >90%
- Model Selection: >85%
- Domain Classification: >95%
- Fuzzing Pass Rate: >85%

### Inference
- Latency: 60-100ms (RTX 3090, BF16)
- Throughput: 10-20 queries/sec
- GPU Memory: 8-12GB (4-bit)

## Production Readiness

- [x] No hardcoded paths (all relative or configurable)
- [x] Environment variable support
- [x] Error handling throughout
- [x] Logging and monitoring hooks
- [x] Health checks
- [x] Graceful shutdown
- [x] Restart recovery
- [x] Input validation
- [x] Output validation
- [x] Type safety

## Testing Status

- [x] Schema validation
- [x] Routing physics logic
- [x] Data generation
- [x] Import checks
- [x] Dependency checks
- [x] File structure verification

## Known Limitations

1. **Single inference thread**: API server processes one request at a time (by design for simplicity)
2. **No batch processing**: Each query is processed individually
3. **No auto-scaling**: Manual deployment management
4. **English-only training**: Multilingual routing not trained
5. **Fixed model set**: 4 target models hardcoded

## Future Enhancements (Not Implemented)

- Multi-GPU inference
- Batch processing mode
- Real-time analytics
- Auto-scaling
- Ray Serve integration
- A/B testing framework
- Model distillation to 1B
- Prometheus metrics
- Grafana dashboards
- Integration tests
- Docker compose setup
- Kubernetes manifests

## Compliance

- [x] Apache 2.0 License
- [x] No proprietary dependencies
- [x] Open source base model (Qwen2.5-3B)
- [x] No telemetry or data collection
- [x] Offline capable

## Author

**Swarm & Bee**
- Website: https://swarmandbee.com
- Tagline: Last mile intelligence

## Signature

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SwarmRouter-v1 | Production Ready | February 24, 2026
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
