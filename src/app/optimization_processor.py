"""
Optimization Processor
Main entry point for running optimization playbooks from YAML configuration files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any

from app.playbooks.optimization_playbooks.tax_efficient_portfolio_transition import TaxEfficientPortfolioTransition
from app.playbooks.optimization_playbooks.base_playbook import GenericPlaybook


PLAYBOOK_TYPES = {
    'generic_optimization': GenericPlaybook,
    'tax_efficient_portfolio_transition': TaxEfficientPortfolioTransition
}

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def optimization_runner(config_path: str) -> Dict[str, Any]:
    """Main runner for optimization playbooks."""
    config = load_config(config_path)
    playbook_type = config.get('playbook_type')

    if not playbook_type:
        raise ValueError("Missing required config key `playbook_type` in `src/app/optimization_processor.py`")

    playbook_cls = PLAYBOOK_TYPES.get(playbook_type)
    if playbook_cls is None:
        valid = ", ".join(PLAYBOOK_TYPES.keys())
        raise ValueError(f"Unknown playbook_type {playbook_type!r}. Valid options: {valid}")

    # Instantiate if a class, otherwise use the provided instance
    playbook = playbook_cls(config) if isinstance(playbook_cls, type) else playbook_cls

    result = playbook.execute()
    return result