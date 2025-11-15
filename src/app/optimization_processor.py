"""
Optimization Processor
Main entry point for running optimization playbooks from YAML configuration files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any

from app.optimization_playbooks.tax_efficient_portfolio_transition import TaxEfficientPortfolioTransition
from app.optimization_playbooks.product_mix_playbook import ProductMixPlaybook


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


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
    """Main runner for optimization playbooks."""
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
        'playbook_type': config.get('playbook_type', 'tax_efficient_portfolio_transition'),
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
        'metadata': config.get('metadata', {}),
        'input_data_path': config.get('input_data_path'),
        'holdings_file': config.get('holdings_file'),
        'purchase_history_file': config.get('purchase_history_file')
    }
    # Create playbook based on type
    playbook_type = config.get('playbook_type', 'tax_efficient_portfolio_transition')

    if playbook_type == 'tax_efficient_portfolio_transition':
        playbook = TaxEfficientPortfolioTransition(config=playbook_config)
    elif playbook_type == 'product_mix':
        playbook = ProductMixPlaybook(config=playbook_config)
    # Add more playbooks here as needed
    else:
        raise ValueError(f"Unknown playbook type: {playbook_type}")

    # Execute playbook (data loading happens inside execute())
    result = playbook.execute()

    return result