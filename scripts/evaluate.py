#!/usr/bin/env python3
"""Evaluation script for conditional text generation."""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List

import torch
import json
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from conditional_text_generation.models import GPT2ConditionalGenerator
from conditional_text_generation.evaluation import TextEvaluator
from conditional_text_generation.utils import set_seed, get_device, load_config


def evaluate_model(
    model: Any,
    evaluator: TextEvaluator,
    test_data: List[Dict[str, str]],
    config: Dict[str, Any],
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on test data.
    
    Args:
        model: Conditional text generator model.
        evaluator: Text evaluator.
        test_data: Test dataset.
        config: Configuration.
        device: Device for computation.
        
    Returns:
        Evaluation metrics.
    """
    predictions = []
    references = []
    conditions = []
    
    print("Generating predictions...")
    for i, item in enumerate(test_data):
        if i % 10 == 0:
            print(f"Processing item {i}/{len(test_data)}")
            
        prompt = item["prompt"]
        condition = item["condition"]
        reference = item.get("reference", "")
        
        # Generate text
        generated_texts = model.generate(
            prompt=prompt,
            condition=condition,
            max_length=config["generation"]["max_length"],
            temperature=config["generation"]["temperature"],
            top_p=config["generation"]["top_p"],
            top_k=config["generation"]["top_k"],
            num_return_sequences=1,
            do_sample=config["generation"]["do_sample"],
            repetition_penalty=config["generation"]["repetition_penalty"],
            no_repeat_ngram_size=config["generation"]["no_repeat_ngram_size"],
        )
        
        predictions.append(generated_texts[0])
        references.append(reference)
        conditions.append(condition)
        
    print("Computing evaluation metrics...")
    
    # Evaluate predictions
    metrics = evaluator.evaluate(
        predictions=predictions,
        references=references if any(ref for ref in references) else None,
        conditions=conditions,
        model=model.model,
        device=device,
    )
    
    # Evaluate conditioning
    conditioning_metrics = evaluator.evaluate_conditioning(
        predictions=predictions,
        conditions=conditions,
    )
    
    metrics.update(conditioning_metrics)
    
    return metrics


def main() -> None:
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate conditional text generation model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file path")
    parser.add_argument("--model_name", type=str, help="Model name to evaluate")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data file")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.model_name:
        config["model"]["name"] = args.model_name
        
    # Set device
    device = get_device()
    config["model"]["device"] = str(device)
    
    # Load test data
    with open(args.test_data, "r") as f:
        test_data = json.load(f)
        
    print(f"Loaded {len(test_data)} test samples")
    
    # Initialize model
    model = GPT2ConditionalGenerator(
        model_name=config["model"]["name"],
        device=device,
    )
    
    # Initialize evaluator
    evaluator = TextEvaluator(tokenizer=model.tokenizer)
    
    # Evaluate model
    metrics = evaluate_model(model, evaluator, test_data, config, device)
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
            
    # Save results
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"\nResults saved to {args.output}")
    
    # Create summary table
    df = pd.DataFrame([metrics])
    df.to_csv(args.output.replace(".json", ".csv"), index=False)
    print(f"Summary table saved to {args.output.replace('.json', '.csv')}")


if __name__ == "__main__":
    main()
