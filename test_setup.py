#!/usr/bin/env python3
"""
Quick sanity check for SwarmRouter setup.
Run this before training to verify everything is working.
"""

import sys
from pathlib import Path

print("=" * 70)
print("SwarmRouter Setup Test")
print("=" * 70)

# Test 1: Import core modules
print("\n1. Testing core imports...")
try:
    from swarmrouter import RouterOutput, apply_routing_physics
    print("   ✓ swarmrouter package imports OK")
except Exception as e:
    print(f"   ✗ Failed to import swarmrouter: {e}")
    sys.exit(1)

# Test 2: Test schema validation
print("\n2. Testing RouterOutput schema...")
try:
    test_routing = {
        "domain": "medical",
        "task_type": "qa",
        "complexity": "medium",
        "risk_level": "high",
        "latency_tier": "standard",
        "cost_sensitivity": "medium",
        "recommended_model": "med-14b",
        "escalation_allowed": True,
        "requires_tools": [],
        "reasoning": "Medical query requires specialist model"
    }
    output = RouterOutput(**test_routing)
    print(f"   ✓ Schema validation OK")
    print(f"     Domain: {output.domain}")
    print(f"     Model: {output.recommended_model}")
except Exception as e:
    print(f"   ✗ Schema validation failed: {e}")
    sys.exit(1)

# Test 3: Test routing physics
print("\n3. Testing routing physics...")
try:
    routing = apply_routing_physics(
        domain="medical",
        user_message="What are the symptoms of pneumonia?",
        task_type="qa",
        complexity="medium"
    )
    assert routing["recommended_model"] == "med-14b", "Medical should route to med-14b"
    print(f"   ✓ Routing physics OK")
    print(f"     Medical query → {routing['recommended_model']}")

    routing2 = apply_routing_physics(
        domain="general",
        user_message="What is the capital of France?",
        task_type="qa",
        complexity="low"
    )
    assert routing2["recommended_model"] == "router-3b", "Simple general should route to router-3b"
    print(f"     General query → {routing2['recommended_model']}")
except Exception as e:
    print(f"   ✗ Routing physics failed: {e}")
    sys.exit(1)

# Test 4: Check data files
print("\n4. Checking data files...")
data_dir = Path(__file__).parent / "data"
topics_dir = data_dir / "topics"
templates_file = data_dir / "templates.json"

expected_topics = ["medical", "aviation", "cre", "compute", "research", "coding", "operations", "general"]
missing_topics = []

for topic in expected_topics:
    topic_file = topics_dir / f"{topic}.txt"
    if not topic_file.exists():
        missing_topics.append(topic)

if missing_topics:
    print(f"   ✗ Missing topic files: {', '.join(missing_topics)}")
    sys.exit(1)

if not templates_file.exists():
    print(f"   ✗ Missing templates.json")
    sys.exit(1)

print(f"   ✓ All data files present")
print(f"     Topics: {len(expected_topics)} files")
print(f"     Templates: OK")

# Test 5: Test data generator
print("\n5. Testing data generator...")
try:
    from data.generator import DatasetGenerator

    generator = DatasetGenerator(data_dir)

    # Generate one example
    example = generator.generate_example(0)
    assert "messages" in example, "Example missing messages"
    assert len(example["messages"]) == 3, "Should have 3 messages (system, user, assistant)"
    assert example["messages"][0]["role"] == "system"
    assert example["messages"][1]["role"] == "user"
    assert example["messages"][2]["role"] == "assistant"

    print(f"   ✓ Data generator OK")
    print(f"     Generated example for domain: {example['metadata']['domain']}")
except Exception as e:
    print(f"   ✗ Data generator failed: {e}")
    sys.exit(1)

# Test 6: Check dependencies
print("\n6. Checking critical dependencies...")
try:
    import torch
    print(f"   ✓ PyTorch {torch.__version__}")

    import transformers
    print(f"   ✓ Transformers {transformers.__version__}")

    import datasets
    print(f"   ✓ Datasets {datasets.__version__}")

    import trl
    print(f"   ✓ TRL {trl.__version__}")

    try:
        import unsloth
        print(f"   ✓ Unsloth OK")
    except:
        print(f"   ⚠ Unsloth not installed (required for training)")

    import fastapi
    print(f"   ✓ FastAPI {fastapi.__version__}")

    import pydantic
    print(f"   ✓ Pydantic {pydantic.__version__}")

except ImportError as e:
    print(f"   ✗ Missing dependency: {e}")
    print(f"\n   Install with: pip install -r requirements.txt")
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ All tests passed! SwarmRouter is ready.")
print("=" * 70)
print("\nNext steps:")
print("  1. Generate dataset:  python data/make_dataset.py")
print("  2. Train model:       python scripts/train.py")
print("  3. Evaluate:          python scripts/eval.py")
print("  4. Try CLI:           python scripts/cli_router.py")
