"""
Error Analysis Script for T5 Text-to-SQL Model
Compares ground truth SQL with model predictions and identifies error patterns.
"""

import os
import json
from collections import defaultdict

def load_lines(path):
    """Load lines from a file."""
    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    return lines

def analyze_errors(nl_queries, gt_sqls, pred_sqls):
    """
    Analyze errors by comparing ground truth and predicted SQL queries.
    Returns a list of error examples.
    """
    errors = []
    correct = []
    
    for i, (nl, gt, pred) in enumerate(zip(nl_queries, gt_sqls, pred_sqls)):
        is_correct = (gt.strip() == pred.strip())
        
        example = {
            'index': i,
            'nl_query': nl,
            'gt_sql': gt,
            'pred_sql': pred,
            'is_correct': is_correct
        }
        
        if is_correct:
            correct.append(example)
        else:
            errors.append(example)
    
    return errors, correct

def categorize_errors(errors):
    """
    Automatically categorize errors into common types.
    This is a simple heuristic - you may need to manually verify.
    """
    categories = defaultdict(list)
    
    for error in errors:
        gt = error['gt_sql'].lower()
        pred = error['pred_sql'].lower()
        
        # Check for various error types
        error_types = []
        
        # Missing JOIN
        if 'join' in gt and 'join' not in pred:
            error_types.append('missing_join')
        
        # Wrong/missing aggregation
        if any(agg in gt for agg in ['count', 'sum', 'avg', 'max', 'min']) and \
           not any(agg in pred for agg in ['count', 'sum', 'avg', 'max', 'min']):
            error_types.append('missing_aggregation')
        
        # Missing WHERE clause
        if 'where' in gt and 'where' not in pred:
            error_types.append('missing_where')
        
        # Missing GROUP BY
        if 'group by' in gt and 'group by' not in pred:
            error_types.append('missing_group_by')
        
        # Missing ORDER BY
        if 'order by' in gt and 'order by' not in pred:
            error_types.append('missing_order_by')
        
        # Wrong table name
        gt_tables = set([t for t in ['flight', 'airline', 'airport'] if t in gt])
        pred_tables = set([t for t in ['flight', 'airline', 'airport'] if t in pred])
        if gt_tables != pred_tables:
            error_types.append('wrong_table')
        
        # Completely wrong structure
        if len(pred.split()) < 3 or 'select' not in pred:
            error_types.append('syntax_error')
        
        # If no specific error identified
        if not error_types:
            error_types.append('other')
        
        for error_type in error_types:
            categories[error_type].append(error)
    
    return categories

def print_analysis_report(errors, correct, error_categories):
    """
    Print a comprehensive analysis report that can be copy-pasted.
    """
    total = len(errors) + len(correct)
    
    print("="*80)
    print("ERROR ANALYSIS REPORT FOR T5 TEXT-TO-SQL MODEL")
    print("="*80)
    print(f"\nTotal examples: {total}")
    print(f"Correct predictions: {len(correct)} ({len(correct)/total*100:.2f}%)")
    print(f"Incorrect predictions: {len(errors)} ({len(errors)/total*100:.2f}%)")
    print("\n" + "="*80)
    
    # Print error category summary
    print("\nERROR CATEGORY SUMMARY:")
    print("-"*80)
    for category, examples in sorted(error_categories.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"{category.upper().replace('_', ' ')}: {len(examples)} occurrences ({len(examples)/len(errors)*100:.1f}% of errors)")
    
    # Print detailed examples for each category
    print("\n" + "="*80)
    print("DETAILED ERROR EXAMPLES (Top 3 examples per category)")
    print("="*80)
    
    for category, examples in sorted(error_categories.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"\n{'='*80}")
        print(f"ERROR TYPE: {category.upper().replace('_', ' ')}")
        print(f"COUNT: {len(examples)}/{total} ({len(examples)/total*100:.2f}%)")
        print(f"{'='*80}")
        
        # Show top 3 examples
        for i, example in enumerate(examples[:3], 1):
            print(f"\nExample {i}:")
            print(f"Natural Language: {example['nl_query']}")
            print(f"Ground Truth SQL: {example['gt_sql']}")
            print(f"Predicted SQL:    {example['pred_sql']}")
            print("-"*80)
    
    # Print all errors in JSON format for easy processing
    print("\n" + "="*80)
    print("ALL ERRORS IN JSON FORMAT (Copy this section to share)")
    print("="*80)
    print("\n```json")
    error_data = {
        'total_examples': total,
        'correct': len(correct),
        'errors': len(errors),
        'error_categories': {cat: len(examples) for cat, examples in error_categories.items()},
        'sample_errors': []
    }
    
    # Add sample errors from each category
    for category, examples in error_categories.items():
        for example in examples[:2]:  # 2 examples per category
            error_data['sample_errors'].append({
                'category': category,
                'nl': example['nl_query'],
                'gt_sql': example['gt_sql'],
                'pred_sql': example['pred_sql']
            })
    
    print(json.dumps(error_data, indent=2))
    print("```")
    
    print("\n" + "="*80)
    print("COPY THE SECTIONS ABOVE AND SEND TO CLAUDE FOR TABLE 5 CREATION")
    print("="*80)

def main():
    """Main function to run error analysis."""
    
    # Define paths
    data_folder = 'data'
    results_folder = 'results'
    
    nl_path = os.path.join(data_folder, 'dev.nl')
    gt_sql_path = os.path.join(data_folder, 'dev.sql')
    pred_sql_path = os.path.join(results_folder, 't5_ft_final_model_dev.sql')
    
    # Check if files exist
    if not os.path.exists(nl_path):
        print(f"Error: {nl_path} not found!")
        print("Please run this script from the part-2-code directory.")
        return
    
    if not os.path.exists(pred_sql_path):
        print(f"Error: {pred_sql_path} not found!")
        print("Please ensure your model has generated predictions.")
        return
    
    # Load data
    print("Loading data...")
    nl_queries = load_lines(nl_path)
    gt_sqls = load_lines(gt_sql_path)
    pred_sqls = load_lines(pred_sql_path)
    
    print(f"Loaded {len(nl_queries)} natural language queries")
    print(f"Loaded {len(gt_sqls)} ground truth SQL queries")
    print(f"Loaded {len(pred_sqls)} predicted SQL queries")
    
    # Analyze errors
    print("\nAnalyzing errors...")
    errors, correct = analyze_errors(nl_queries, gt_sqls, pred_sqls)
    
    # Categorize errors
    print("Categorizing errors...")
    error_categories = categorize_errors(errors)
    
    # Print comprehensive report
    print_analysis_report(errors, correct, error_categories)
    
    # Save to file for easy access
    output_file = 'error_analysis_report.txt'
    print(f"\n\nSaving report to {output_file}...")
    
    import sys
    original_stdout = sys.stdout
    with open(output_file, 'w') as f:
        sys.stdout = f
        print_analysis_report(errors, correct, error_categories)
    sys.stdout = original_stdout
    
    print(f"Report saved! You can also view it with: cat {output_file}")

if __name__ == "__main__":
    main()