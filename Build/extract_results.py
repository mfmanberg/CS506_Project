"""
Extract model results from executed notebooks and log them.
"""
import json
from pathlib import Path
from datetime import datetime
import re


def extract_notebook_outputs(notebook_path):
    """Extract output text and metrics from a notebook."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    results = {
        'notebook': notebook_path.name,
        'cells_executed': 0,
        'total_cells': 0,
        'metrics': {},
        'outputs': []
    }
    
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            results['total_cells'] += 1
            
            # Check if cell was executed (either by execution_count or presence of outputs)
            execution_count = cell.get('execution_count')
            outputs = cell.get('outputs', [])
            
            # Cell is considered executed if it has execution_count OR has outputs
            if (execution_count is not None and execution_count > 0) or len(outputs) > 0:
                results['cells_executed'] += 1
            
            # Extract outputs
            for output in outputs:
                # Get text output
                if 'text' in output:
                    text = ''.join(output['text']) if isinstance(output['text'], list) else output['text']
                    results['outputs'].append(text)
                
                # Get stdout
                if output.get('name') == 'stdout' and 'text' in output:
                    text = ''.join(output['text']) if isinstance(output['text'], list) else output['text']
                    results['outputs'].append(text)
    
    # Extract metrics from outputs
    output_text = '\n'.join(results['outputs'])
    
    # Common metric patterns
    patterns = {
        'MSE': r'(?:MSE|Mean Squared Error)[\s:=]+([0-9.]+(?:e[+-]?[0-9]+)?)',
        'RMSE': r'(?:RMSE|Root Mean Squared Error)[\s:=]+([0-9.]+(?:e[+-]?[0-9]+)?)',
        'R2': r'(?:R\^?2|R-squared|R2 Score)[\s:=]+([0-9.-]+)',
        'MAE': r'(?:MAE|Mean Absolute Error)[\s:=]+([0-9.]+(?:e[+-]?[0-9]+)?)',
        'Accuracy': r'(?:Accuracy|Score)[\s:=]+([0-9.]+)%?',
    }
    
    for metric_name, pattern in patterns.items():
        matches = re.findall(pattern, output_text, re.IGNORECASE)
        if matches:
            # Take the last occurrence (usually the final result)
            try:
                results['metrics'][metric_name] = float(matches[-1])
            except ValueError:
                results['metrics'][metric_name] = matches[-1]
    
    return results


def format_results_table(all_results):
    """Format results as a markdown table."""
    lines = []
    lines.append("| Notebook | Cells Executed | MSE | RMSE | R² | MAE |")
    lines.append("|----------|---------------|-----|------|----|----|")
    
    for result in all_results:
        name = result['notebook'].replace('.ipynb', '')
        # Show cells with outputs / total (some cells like imports don't produce output)
        cells = f"{result['cells_executed']}/{result['total_cells']}"
        mse = result['metrics'].get('MSE', '-')
        rmse = result['metrics'].get('RMSE', '-')
        r2 = result['metrics'].get('R2', '-')
        mae = result['metrics'].get('MAE', '-')
        
        # Format numbers
        if isinstance(mse, (int, float)):
            mse = f"{mse:.2f}" if mse < 1000 else f"{mse:.2e}"
        if isinstance(rmse, (int, float)):
            rmse = f"{rmse:.2f}" if rmse < 1000 else f"{rmse:.2e}"
        if isinstance(r2, (int, float)):
            r2 = f"{r2:.4f}"
        if isinstance(mae, (int, float)):
            mae = f"{mae:.2f}" if mae < 1000 else f"{mae:.2e}"
        
        lines.append(f"| {name} | {cells} | {mse} | {rmse} | {r2} | {mae} |")
    
    return '\n'.join(lines)


def log_results(all_results, log_file):
    """Append results to the log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create log entry
    log_entry = [
        "",
        "=" * 80,
        f"Execution Results - {timestamp}",
        "=" * 80,
        "",
        format_results_table(all_results),
        "",
        "### Detailed Metrics",
        ""
    ]
    
    # Add detailed metrics for each notebook
    for result in all_results:
        log_entry.append(f"**{result['notebook']}**")
        log_entry.append(f"- Cells with outputs: {result['cells_executed']}/{result['total_cells']} (cells like imports/comments may not produce output)")
        if result['metrics']:
            log_entry.append("- Metrics:")
            for metric, value in result['metrics'].items():
                if isinstance(value, float):
                    log_entry.append(f"  - {metric}: {value:.6f}")
                else:
                    log_entry.append(f"  - {metric}: {value}")
        else:
            log_entry.append("- No metrics extracted")
        log_entry.append("")
    
    # Append to log file
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write('\n'.join(log_entry))
    
    print(f"Results logged to: {log_file}")


def main():
    """Extract results from all target notebooks."""
    # Get the Build folder
    build_folder = Path(__file__).parent.resolve()
    project_root = build_folder.parent  # Go up one level to project root
    log_file = build_folder / 'model_results.log'  # Output to Build folder
    
    notebooks = [
        project_root / '2_FIGURES' / '2_data_exploration' / 'nyiso_data_exploration.ipynb',
        project_root / '3_OUTPUT' / '3_linear_regression' / 'linear_regression.ipynb',
        project_root / '3_OUTPUT' / '3_svr' / 'SVM_Trunc.ipynb',
        project_root / '3_OUTPUT' / '3_svr' / 'SVMDaily.ipynb',
        project_root / '3_OUTPUT' / '3_svr' / 'SVMDailywoutMeso.ipynb',
        project_root / '3_OUTPUT' / '3_svr' / 'SVMHourly.ipynb',
        project_root / '3_OUTPUT' / '3_xg_boost' / 'ComparisonMetrics.ipynb',
        project_root / '3_OUTPUT' / '3_xg_boost' / 'XGBoost_PostMid.ipynb',
        project_root / '3_OUTPUT' / '3_xg_boost' / 'XGBoost_Testing.ipynb',
    ]
    
    print("Extracting results from notebooks...")
    all_results = []
    
    for nb_path in notebooks:
        if nb_path.exists():
            print(f"  Processing: {nb_path.name}")
            results = extract_notebook_outputs(nb_path)
            all_results.append(results)
        else:
            print(f"  Warning: Not found - {nb_path.name}")
    
    # Log the results
    log_results(all_results, log_file)
    
    # Also print summary to console
    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    print(format_results_table(all_results))
    print("=" * 80)
    print(f"\nFull log saved to: {log_file}")


if __name__ == '__main__':
    main()
