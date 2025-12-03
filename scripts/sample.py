#!/usr/bin/env python3
"""Sampling script for conditional text generation."""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

import torch
import json
from transformers import GPT2Tokenizer

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from conditional_text_generation.models import GPT2ConditionalGenerator, AutoConditionalGenerator
from conditional_text_generation.utils import set_seed, get_device, load_config


def generate_samples(
    model: Any,
    prompts: List[str],
    conditions: List[str],
    config: Dict[str, Any],
    output_file: str,
) -> None:
    """Generate samples and save to file.
    
    Args:
        model: Conditional text generator model.
        prompts: List of prompts.
        conditions: List of conditions.
        config: Generation configuration.
        output_file: Output file path.
    """
    results = []
    
    for prompt, condition in zip(prompts, conditions):
        print(f"Generating for condition: {condition}")
        print(f"Prompt: {prompt}")
        
        # Generate text
        generated_texts = model.generate(
            prompt=prompt,
            condition=condition,
            max_length=config["generation"]["max_length"],
            temperature=config["generation"]["temperature"],
            top_p=config["generation"]["top_p"],
            top_k=config["generation"]["top_k"],
            num_return_sequences=config["generation"]["num_return_sequences"],
            do_sample=config["generation"]["do_sample"],
            repetition_penalty=config["generation"]["repetition_penalty"],
            no_repeat_ngram_size=config["generation"]["no_repeat_ngram_size"],
        )
        
        result = {
            "prompt": prompt,
            "condition": condition,
            "generated_texts": generated_texts,
        }
        results.append(result)
        
        # Print results
        for i, text in enumerate(generated_texts):
            print(f"Generated {i+1}: {text}")
        print("-" * 50)
        
    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Results saved to {output_file}")


def main() -> None:
    """Main sampling function."""
    parser = argparse.ArgumentParser(description="Generate conditional text samples")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file path")
    parser.add_argument("--model_name", type=str, help="Model name to use")
    parser.add_argument("--prompts", type=str, nargs="+", help="Prompts to generate from")
    parser.add_argument("--conditions", type=str, nargs="+", help="Conditions for generation")
    parser.add_argument("--prompt_file", type=str, help="File containing prompts and conditions")
    parser.add_argument("--output", type=str, default="samples.json", help="Output file")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples per prompt")
    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, help="Top-p sampling parameter")
    parser.add_argument("--max_length", type=int, help="Maximum generation length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.model_name:
        config["model"]["name"] = args.model_name
    if args.temperature:
        config["generation"]["temperature"] = args.temperature
    if args.top_p:
        config["generation"]["top_p"] = args.top_p
    if args.max_length:
        config["generation"]["max_length"] = args.max_length
    if args.num_samples:
        config["generation"]["num_return_sequences"] = args.num_samples
        
    # Set device
    device = get_device()
    config["model"]["device"] = str(device)
    
    # Initialize model
    if config["model"]["name"].startswith("gpt2"):
        model = GPT2ConditionalGenerator(
            model_name=config["model"]["name"],
            device=device,
        )
    else:
        model = AutoConditionalGenerator(
            model_name=config["model"]["name"],
            device=device,
            trust_remote_code=config["model"]["trust_remote_code"],
        )
    
    # Prepare prompts and conditions
    if args.prompt_file:
        with open(args.prompt_file, "r") as f:
            data = json.load(f)
        prompts = data["prompts"]
        conditions = data["conditions"]
    elif args.prompts and args.conditions:
        prompts = args.prompts
        conditions = args.conditions
    else:
        # Default examples
        prompts = [
            "I really love this product because",
            "The weather today is",
            "Once upon a time",
            "The recipe calls for",
            "I think this movie is",
        ]
        conditions = [
            "positive review",
            "weather description",
            "story beginning",
            "cooking instructions",
            "movie review",
        ]
    
    # Ensure equal length
    min_length = min(len(prompts), len(conditions))
    prompts = prompts[:min_length]
    conditions = conditions[:min_length]
    
    # Generate samples
    generate_samples(model, prompts, conditions, config, args.output)


if __name__ == "__main__":
    main()
