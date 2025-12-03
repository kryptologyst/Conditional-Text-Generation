"""Streamlit demo for conditional text generation."""

import streamlit as st
import sys
from pathlib import Path
from typing import List, Dict, Any
import json
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from conditional_text_generation.models import GPT2ConditionalGenerator, AutoConditionalGenerator
from conditional_text_generation.evaluation import TextEvaluator
from conditional_text_generation.utils import set_seed, get_device, load_config


@st.cache_resource
def load_model(model_name: str, device: str) -> Any:
    """Load and cache the model.
    
    Args:
        model_name: Name of the model to load.
        device: Device to run the model on.
        
    Returns:
        Loaded model.
    """
    if model_name.startswith("gpt2"):
        return GPT2ConditionalGenerator(model_name=model_name, device=device)
    else:
        return AutoConditionalGenerator(model_name=model_name, device=device)


def main() -> None:
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Conditional Text Generation",
        page_icon="📝",
        layout="wide",
    )
    
    st.title("📝 Conditional Text Generation")
    st.markdown("Generate text conditioned on specific prompts and contexts using transformer models.")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Model selection
    model_options = {
        "GPT-2": "gpt2",
        "GPT-2 Medium": "gpt2-medium",
        "GPT-2 Large": "gpt2-large",
        "GPT-2 XL": "gpt2-xl",
    }
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=list(model_options.keys()),
        index=0,
    )
    model_name = model_options[selected_model]
    
    # Device selection
    device = get_device()
    st.sidebar.info(f"Using device: {device}")
    
    # Load model
    with st.spinner(f"Loading {selected_model}..."):
        model = load_model(model_name, device)
    
    st.sidebar.success(f"✅ {selected_model} loaded successfully!")
    
    # Generation parameters
    st.sidebar.header("Generation Parameters")
    
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Controls randomness. Lower values make output more deterministic.",
    )
    
    top_p = st.sidebar.slider(
        "Top-p (Nucleus Sampling)",
        min_value=0.1,
        max_value=1.0,
        value=0.9,
        step=0.05,
        help="Controls diversity by considering only the top-p probability mass.",
    )
    
    top_k = st.sidebar.slider(
        "Top-k",
        min_value=1,
        max_value=100,
        value=50,
        step=1,
        help="Controls diversity by considering only the top-k tokens.",
    )
    
    max_length = st.sidebar.slider(
        "Max Length",
        min_value=20,
        max_value=200,
        value=100,
        step=10,
        help="Maximum length of generated text.",
    )
    
    num_sequences = st.sidebar.slider(
        "Number of Sequences",
        min_value=1,
        max_value=5,
        value=1,
        step=1,
        help="Number of different sequences to generate.",
    )
    
    repetition_penalty = st.sidebar.slider(
        "Repetition Penalty",
        min_value=1.0,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Penalty for repeating tokens.",
    )
    
    # Seed control
    use_seed = st.sidebar.checkbox("Use Fixed Seed", value=False)
    if use_seed:
        seed = st.sidebar.number_input("Seed", min_value=0, max_value=1000000, value=42)
        set_seed(seed)
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Input")
        
        # Condition input
        condition = st.text_input(
            "Condition",
            value="positive review",
            help="The condition or context for text generation (e.g., 'positive review', 'story beginning')",
        )
        
        # Prompt input
        prompt = st.text_area(
            "Prompt",
            value="I really love this product because",
            height=100,
            help="The starting text for generation",
        )
        
        # Generate button
        if st.button("🚀 Generate Text", type="primary"):
            if not condition.strip() or not prompt.strip():
                st.error("Please provide both condition and prompt.")
            else:
                with st.spinner("Generating text..."):
                    try:
                        generated_texts = model.generate(
                            prompt=prompt,
                            condition=condition,
                            max_length=max_length,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            num_return_sequences=num_sequences,
                            do_sample=True,
                            repetition_penalty=repetition_penalty,
                        )
                        
                        # Store results in session state
                        st.session_state.generated_texts = generated_texts
                        st.session_state.condition = condition
                        st.session_state.prompt = prompt
                        
                    except Exception as e:
                        st.error(f"Error generating text: {str(e)}")
    
    with col2:
        st.header("Output")
        
        if "generated_texts" in st.session_state:
            st.subheader(f"Generated Text ({st.session_state.condition})")
            
            for i, text in enumerate(st.session_state.generated_texts):
                with st.expander(f"Sequence {i+1}", expanded=True):
                    st.write(text)
                    
                    # Copy button
                    if st.button(f"📋 Copy Sequence {i+1}"):
                        st.write("Copied to clipboard!")
            
            # Evaluation metrics
            if st.checkbox("Show Evaluation Metrics"):
                st.subheader("Evaluation Metrics")
                
                evaluator = TextEvaluator(tokenizer=model.tokenizer)
                
                # Calculate metrics for generated texts
                metrics = evaluator.evaluate(
                    predictions=st.session_state.generated_texts,
                    conditions=[st.session_state.condition] * len(st.session_state.generated_texts),
                )
                
                # Display metrics
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.metric("Average Length", f"{metrics['avg_length']:.1f}")
                    st.metric("Distinct-2", f"{metrics['distinct_2']:.3f}")
                    st.metric("Distinct-3", f"{metrics['distinct_3']:.3f}")
                
                with col_b:
                    st.metric("Self-BLEU", f"{metrics['self_bleu']:.3f}")
                    st.metric("Repetition Ratio", f"{metrics['repetition_ratio']:.3f}")
                    st.metric("Conditioning Accuracy", f"{metrics.get('conditioning_accuracy', 0.0):.3f}")
        else:
            st.info("Click 'Generate Text' to see results here.")
    
    # Examples section
    st.header("📚 Examples")
    
    example_tabs = st.tabs(["Reviews", "Stories", "Descriptions", "Instructions"])
    
    with example_tabs[0]:
        st.subheader("Product Reviews")
        examples = [
            ("positive review", "This product is amazing because"),
            ("negative review", "I was disappointed with this because"),
            ("neutral review", "This product is okay because"),
        ]
        
        for condition, prompt in examples:
            if st.button(f"Use: {condition}"):
                st.session_state.example_condition = condition
                st.session_state.example_prompt = prompt
                st.rerun()
    
    with example_tabs[1]:
        st.subheader("Story Beginnings")
        examples = [
            ("story beginning", "Once upon a time"),
            ("horror story", "In the dark forest"),
            ("romance story", "She looked into his eyes"),
        ]
        
        for condition, prompt in examples:
            if st.button(f"Use: {condition}"):
                st.session_state.example_condition = condition
                st.session_state.example_prompt = prompt
                st.rerun()
    
    with example_tabs[2]:
        st.subheader("Descriptions")
        examples = [
            ("weather description", "The weather today is"),
            ("nature description", "The sunset over the ocean"),
            ("character description", "She walked into the room"),
        ]
        
        for condition, prompt in examples:
            if st.button(f"Use: {condition}"):
                st.session_state.example_condition = condition
                st.session_state.example_prompt = prompt
                st.rerun()
    
    with example_tabs[3]:
        st.subheader("Instructions")
        examples = [
            ("cooking instructions", "To make this recipe"),
            ("technical instructions", "To fix this problem"),
            ("exercise instructions", "To perform this exercise"),
        ]
        
        for condition, prompt in examples:
            if st.button(f"Use: {condition}"):
                st.session_state.example_condition = condition
                st.session_state.example_prompt = prompt
                st.rerun()
    
    # Handle example selection
    if "example_condition" in st.session_state and "example_prompt" in st.session_state:
        st.session_state.condition = st.session_state.example_condition
        st.session_state.prompt = st.session_state.example_prompt
        del st.session_state.example_condition
        del st.session_state.example_prompt
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with [Streamlit](https://streamlit.io/) and [Transformers](https://huggingface.co/transformers/)"
    )


if __name__ == "__main__":
    main()
