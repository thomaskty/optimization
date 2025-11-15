import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from app.optimization_processor import optimization_runner

optimization_runner('tests/test_data/sample_data/portfolio_optimization/config_tax_loss_harvesting.yaml')
