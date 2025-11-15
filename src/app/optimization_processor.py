"""
Optimization Processor
Main entry point for running optimization optimization_playbooks from YAML configuration files.
"""

import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import json

from app.optimization_playbooks.portfolio_optimization_playbook import PortfolioOptimizationPlaybook


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_data_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Load input data based on configuration."""
    input_data = {}

    # New format: datasets section
    if 'datasets' in config:
        datasets = config['datasets']
        if 'holdings' in datasets:
            input_data['holdings'] = pd.read_csv(datasets['holdings'])
        if 'acquisitions' in datasets:
            input_data['purchase_history'] = pd.read_csv(datasets['acquisitions'])

    # Legacy format: input_data_path + individual files
    elif 'input_data_path' in config:
        data_path = Path(config.get('input_data_path', ''))
        if 'holdings_file' in config:
            input_data['holdings'] = pd.read_csv(data_path / config['holdings_file'])
        if 'purchase_history_file' in config:
            input_data['purchase_history'] = pd.read_csv(data_path / config['purchase_history_file'])

    return input_data


def get_output_directory(config: Dict[str, Any]) -> str:
    """Construct output directory path from config."""
    model_name = config.get('model_name') or config.get('name', 'optimization')
    model_id = config.get('model_id')
    base_path = config.get('base_path', '.')

    # Check for explicit output directory
    if 'output' in config and isinstance(config['output'], dict):
        output_dir = config['output'].get('directory')
        if output_dir is not None:
            if Path(output_dir).is_absolute():
                base_output = Path(output_dir)
            else:
                base_output = Path(base_path) / output_dir
            # Add model_id subdirectory if available
            if model_id:
                return str(base_output / model_id)
            return str(base_output)

    if 'output_dir' in config:
        output_dir = config.get('output_dir')
        if output_dir is not None:
            if Path(output_dir).is_absolute():
                base_output = Path(output_dir)
            else:
                base_output = Path(base_path) / output_dir
            # Add model_id subdirectory if available
            if model_id:
                return str(base_output / model_id)
            return str(base_output)

    # Default: base_path/outputs/model_name/model_id
    base_output = Path(base_path) / 'outputs' / model_name
    if model_id:
        return str(base_output / model_id)
    return str(base_output)


def optimization_runner(config_path: str) -> Dict[str, Any]:
    """Main runner for optimization optimization_playbooks."""
    # Load configuration from YAML
    config = load_config(config_path)

    # Get output directory
    output_dir = get_output_directory(config)

    # Prepare config dict for playbook
    playbook_config = {
        'model_name': config.get('model_name', config.get('name', 'optimization')),
        'model_id': config.get('model_id'),
        'submission_id': config.get('submission_id'),
        'model_type': config.get('model_type', config.get('optimizer_type', 'milp')),
        'playbook_type': config.get('playbook_type', 'portfolio_optimization'),
        'datasets': config.get('datasets', {}),
        'columns': config.get('columns', {}),
        'constraints': config.get('constraints', []),
        'objective': config.get('objective', {}),
        'optimizer_params': config.get('optimizer_params', {}),
        'tax_parameters': config.get('tax_parameters', config.get('metadata', {})),
        'output': {
            'save_results': config.get('save_results', True),
            'directory': output_dir,
            'formats': config.get('output', {}).get('formats', ['json', 'csv'])
        },
        'metadata': config.get('metadata', {})
    }

    # Load input data
    input_data = load_data_from_config(config)

    # Create and execute playbook
    playbook_type = config.get('playbook_type', 'portfolio_optimization')

    if playbook_type == 'portfolio_optimization':
        playbook = PortfolioOptimizationPlaybook(config=playbook_config)
    else:
        raise ValueError(f"Unknown playbook type: {playbook_type}")

    # Execute playbook
    result = playbook.execute(input_data)

    # Print result as JSON
    if result.get('status') == 'success' and 'output' in result:
        output = result['output']
        optimization_result = {
            'status': output.get('status', result['status']),
            'sell_decisions': output.get('sell_decisions', []),
            'summary': output.get('optimization_summary', {})
        }
        print("\n" + json.dumps(optimization_result, indent=2, default=str))
    else:
        print("\n" + json.dumps(result, indent=2, default=str))

    return result