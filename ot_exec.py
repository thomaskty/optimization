import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from app.optimization_processor import optimization_runner

# portfolio optimization playbooks
optimization_runner('tests/test_data/sample_data/tax_efficient_portfolio_transition/config.yaml')
# optimization_runner('tests/test_data/sample_data/tax_efficient_portfolio_transition/config_tax_loss_harvesting.yaml')

# product mix playbooks
# optimization_runner('tests/test_data/sample_data/product_mix/config_basic.yaml')
# optimization_runner('tests/test_data/sample_data/product_mix/config_advanced.yaml')


