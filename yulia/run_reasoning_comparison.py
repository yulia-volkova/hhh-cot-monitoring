"""Run evaluation with reasoning ON and OFF, compare results."""

from __future__ import annotations
import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from yulia.evaluate_discrim import run_evaluation


def save_jsonl(data, filepath):
    """Save data as JSONL."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved {len(data)} examples to {filepath}")


def load_jsonl(filepath):
    """Load data from JSONL."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def compare_reasoning_modes(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    num_samples: int = None,
    config: str = "explicit",
    backend: str = "transformers",
    batch_size: int = 128,
    output_dir: str = "results/discrim_eval",
):
    """
    Run evaluation with reasoning ON and OFF, compare answers.
    
    Args:
        model_name: Model to evaluate
        num_samples: Number of samples (None for full dataset)
        config: "explicit" or "implicit"
        backend: "transformers" or "vllm"
        output_dir: Where to save results
    
    Returns:
        Dictionary with comparison statistics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("REASONING ON vs OFF COMPARISON")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Samples: {num_samples if num_samples else 'ALL (full dataset)'}")
    print(f"Config: {config}")
    print(f"Backend: {backend}")
    print("="*80)
    
    # Run with reasoning ON
    print("\n### EVALUATING WITH REASONING ON ###\n")
    results_on_path = run_evaluation(
        model_name,
        output_path / "reasoning_on",
        config=config,
        max_examples=num_samples,
        backend=backend,
        batch_size=batch_size,
        enable_reasoning=True,
        run_name="reasoning_on",
    )
    
    # Load reasoning ON results from JSONL
    results_on = load_jsonl(results_on_path)
    metadata_on_path = results_on_path.parent / "metadata.json"
    with open(metadata_on_path, 'r', encoding='utf-8') as f:
        metadata_on = json.load(f)
    
    # Run with reasoning OFF
    print("\n### EVALUATING WITH REASONING OFF ###\n")
    results_off_path = run_evaluation(
        model_name,
        output_path / "reasoning_off",
        config=config,
        max_examples=num_samples,
        backend=backend,
        batch_size=batch_size,
        enable_reasoning=False,
        run_name="reasoning_off",
    )
    
    # Load reasoning OFF results from JSONL
    results_off = load_jsonl(results_off_path)
    metadata_off_path = results_off_path.parent / "metadata.json"
    with open(metadata_off_path, 'r', encoding='utf-8') as f:
        metadata_off = json.load(f)
    
    # Save as JSONL
    jsonl_on = output_path / "reasoning_on.jsonl"
    jsonl_off = output_path / "reasoning_off.jsonl"
    
    save_jsonl(results_on, jsonl_on)
    save_jsonl(results_off, jsonl_off)
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARING DECISIONS")
    print("="*80)
    
    total = len(results_on)
    same_decision = 0
    different_decision = 0
    changed_samples = []
    
    for i, (r_on, r_off) in enumerate(zip(results_on, results_off)):
        if r_on['decision'] == r_off['decision']:
            same_decision += 1
        else:
            different_decision += 1
            changed_samples.append({
                'index': i,
                'filled_template': r_on['filled_template'],
                'age': r_on.get('age'),
                'gender': r_on.get('gender'),
                'race': r_on.get('race'),
                'decision_with_reasoning': r_on['decision'],
                'decision_without_reasoning': r_off['decision'],
                'reasoning': r_on.get('reasoning', ''),
                'response_with_reasoning': r_on['response'],
                'response_without_reasoning': r_off['response'],
            })
    
    # Calculate statistics
    change_rate = (different_decision / total * 100) if total > 0 else 0
    
    print(f"\nTotal samples: {total}")
    print(f"Same decision: {same_decision} ({same_decision/total*100:.1f}%)")
    print(f"Different decision: {different_decision} ({different_decision/total*100:.1f}%)")
    print(f"\nAnswer change rate: {change_rate:.1f}%")
    print(f"SAMPLES WITH CHANGED ANSWERS: {different_decision}")
    
    # Count Yes/No distribution
    yes_on = sum(1 for r in results_on if r['decision'] == 'Yes')
    no_on = sum(1 for r in results_on if r['decision'] == 'No')
    yes_off = sum(1 for r in results_off if r['decision'] == 'Yes')
    no_off = sum(1 for r in results_off if r['decision'] == 'No')
    
    print(f"\nDecision distribution:")
    print(f"  Reasoning ON:  Yes={yes_on}, No={no_on}")
    print(f"  Reasoning OFF: Yes={yes_off}, No={no_off}")
    
    # Save changed samples
    if changed_samples:
        changed_path = output_path / "samples_with_changed_answers.jsonl"
        save_jsonl(changed_samples, changed_path)
        
        print(f"\nSaved {len(changed_samples)} samples with changed answers to:")
        print(f"   {changed_path}")
        
        # Also save a filtered version of reasoning_on results (for bias analysis)
        changed_indices = {c['index'] for c in changed_samples}
        changed_reasoning_on = [r for i, r in enumerate(results_on) if i in changed_indices]
        
        changed_reasoning_path = output_path / "reasoning_on_changed_only.jsonl"
        save_jsonl(changed_reasoning_on, changed_reasoning_path)
        print(f"\nSaved {len(changed_reasoning_on)} reasoning ON results (changed samples only) to:")
        print(f"   {changed_reasoning_path}")
        print(f"   (This file can be used with bias_analysis.py)")
        
        # Show examples
        print(f"\n{'='*80}")
        print("EXAMPLES OF CHANGED ANSWERS (first 3)")
        print("="*80)
        for i, sample in enumerate(changed_samples[:3]):
            print(f"\nExample {i+1}:")
            print(f"  Demographics: age={sample['age']}, gender={sample['gender']}, race={sample['race']}")
            print(f"  Decision WITH reasoning:    {sample['decision_with_reasoning']}")
            print(f"  Decision WITHOUT reasoning: {sample['decision_without_reasoning']}")
            print(f"  Prompt: {sample['filled_template'][:150]}...")
            print("-"*80)
    else:
        print("\nNo samples with changed answers found.")
    
    # Save summary
    summary = {
        'metadata': {
            'model': model_name,
            'config': config,
            'num_samples': num_samples,
            'backend': backend,
            'reasoning_on_metadata': metadata_on,
            'reasoning_off_metadata': metadata_off,
        },
        'statistics': {
            'total_samples': total,
            'same_decision': same_decision,
            'different_decision': different_decision,
            'change_rate_percent': change_rate,
            'reasoning_on_yes': yes_on,
            'reasoning_on_no': no_on,
            'reasoning_off_yes': yes_off,
            'reasoning_off_no': no_off,
        },
        'samples_with_changes': len(changed_samples),
    }
    
    summary_path = output_path / "comparison_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved comparison summary to: {summary_path}")
    
    # Generate visualization
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Decision agreement pie chart
    labels = ['Same Decision', 'Different Decision']
    sizes = [same_decision, different_decision]
    colors = ['lightgreen', 'coral']
    axes[0, 0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    axes[0, 0].set_title('Decision Agreement')
    
    # Plot 2: Decision distribution (counts)
    categories = ['Yes', 'No']
    on_values = [yes_on, no_on]
    off_values = [yes_off, no_off]
    x = range(len(categories))
    width = 0.35
    axes[0, 1].bar([i - width/2 for i in x], on_values, width, label='Reasoning ON', color='coral')
    axes[0, 1].bar([i + width/2 for i in x], off_values, width, label='Reasoning OFF', color='steelblue')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Decision Distribution')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(categories)
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Decision distribution (percentages)
    on_pcts = [yes_on/total*100, no_on/total*100]
    off_pcts = [yes_off/total*100, no_off/total*100]
    axes[1, 0].bar([i - width/2 for i in x], on_pcts, width, label='Reasoning ON', color='coral')
    axes[1, 0].bar([i + width/2 for i in x], off_pcts, width, label='Reasoning OFF', color='steelblue')
    axes[1, 0].set_ylabel('Percentage (%)')
    axes[1, 0].set_title('Decision Distribution (%)')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(categories)
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: Summary text
    axes[1, 1].axis('off')
    summary_text = f"""
    SUMMARY
    
    Config: {config.upper()}
    Total Examples: {total}
    
    Same Decision: {same_decision}
    Different Decision: {different_decision}
    
    Change Rate: {change_rate:.1f}%
    
    Reasoning ON:
      Yes: {yes_on} ({yes_on/total*100:.1f}%)
      No: {no_on} ({no_on/total*100:.1f}%)
    
    Reasoning OFF:
      Yes: {yes_off} ({yes_off/total*100:.1f}%)
      No: {no_off} ({no_off/total*100:.1f}%)
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                    fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle(f'Reasoning ON vs OFF Comparison ({config}) | Change Rate: {change_rate:.1f}%', 
                 fontsize=14, y=0.995)
    plt.tight_layout()
    
    plot_path = output_path / "reasoning_comparison_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot to: {plot_path}")
    plt.close()
    
    print("\n" + "="*80)
    print(f"SUMMARY: {different_decision} out of {total} samples changed ({change_rate:.1f}%)")
    print("="*80)
    print("FILES SAVED:")
    print("="*80)
    print(f"  {jsonl_on}")
    print(f"  {jsonl_off}")
    if changed_samples:
        print(f"  {changed_path}")
        print(f"  {changed_reasoning_path}")
    print(f"  {summary_path}")
    print(f"  {plot_path}")
    print("="*80)
    print("\nFOR BIAS ANALYSIS, YOU CAN USE:")
    print(f"  1. Full dataset:     python yulia/bias_analysis.py {output_path / 'reasoning_on' / 'discrim_eval_results.jsonl'}")
    if changed_samples:
        print(f"  2. Changed samples:  python yulia/bias_analysis.py {changed_reasoning_path}")
        print(f"     (Only use option 2 if you have enough changed samples for meaningful analysis)")
    print("="*80)
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare reasoning ON vs OFF")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples. Use None or 0 for full dataset. (default: None = all samples)")
    parser.add_argument("--config", default="explicit", choices=["explicit", "implicit"])
    parser.add_argument("--backend", default="transformers", choices=["transformers", "vllm"])
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (128 for A100 80GB)")
    parser.add_argument("--output", default="results/discrim_eval")
    
    args = parser.parse_args()
    
    # Convert 0 to None for "all samples"
    num_samples = args.samples if args.samples and args.samples > 0 else None
    
    compare_reasoning_modes(
        model_name=args.model,
        num_samples=num_samples,
        config=args.config,
        backend=args.backend,
        batch_size=args.batch_size,
        output_dir=args.output,
    )

