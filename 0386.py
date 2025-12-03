#!/usr/bin/env python3
"""
Project 386: Conditional Text Generation - Modernized Implementation

This is a modernized version of the original conditional text generation project.
The original simple implementation has been refactored into a comprehensive,
production-ready package with advanced features, evaluation metrics, and interactive demos.

For the full modern implementation, see the src/ directory and README.md
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from conditional_text_generation.utils import set_seed, get_device


def generate_conditional_text(
    prompt: str, 
    condition: str, 
    max_length: int = 100, 
    temperature: float = 0.7, 
    top_p: float = 0.9,
    device: str = "auto"
) -> str:
    """Generate conditional text using GPT-2.
    
    Args:
        prompt: Input prompt text.
        condition: Conditioning text.
        max_length: Maximum length of generated text.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        device: Device to run on.
        
    Returns:
        Generated text.
    """
    # Set device
    if device == "auto":
        device = get_device()
    else:
        device = torch.device(device)
    
    # Load model and tokenizer
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.to(device)
    model.eval()
    
    # Combine prompt and condition
    conditional_prompt = f"{condition}: {prompt}"
    inputs = tokenizer.encode(conditional_prompt, return_tensors='pt')
    inputs = inputs.to(device)
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            inputs, 
            max_length=max_length, 
            num_return_sequences=1,
            temperature=temperature, 
            top_p=top_p, 
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the original prompt from generated text
    generated_text = generated_text[len(conditional_prompt):].strip()
    
    return generated_text


def main():
    """Main function demonstrating conditional text generation."""
    # Set seed for reproducibility
    set_seed(42)
    
    print("Project 386: Conditional Text Generation - Modernized")
    print("=" * 60)
    
    # Example 1: Product Review
    print("\n1. Product Review Generation:")
    print("-" * 30)
    condition = "Write a positive review for a product"
    prompt = "I really love this product because"
    
    generated_text = generate_conditional_text(prompt, condition)
    print(f"Condition: {condition}")
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    
    # Example 2: Story Beginning
    print("\n2. Story Beginning Generation:")
    print("-" * 30)
    condition = "Write a story beginning"
    prompt = "Once upon a time"
    
    generated_text = generate_conditional_text(prompt, condition)
    print(f"Condition: {condition}")
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    
    # Example 3: Weather Description
    print("\n3. Weather Description Generation:")
    print("-" * 30)
    condition = "Describe the weather"
    prompt = "The weather today is"
    
    generated_text = generate_conditional_text(prompt, condition)
    print(f"Condition: {condition}")
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    
    print("\n" + "=" * 60)
    print("For advanced features, interactive demos, and comprehensive evaluation,")
    print("see the full implementation in the src/ directory and README.md")
    print("=" * 60)


if __name__ == "__main__":
    main()