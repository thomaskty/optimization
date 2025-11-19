import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from app.optimization_processor import optimization_runner

# portfolio optimization playbooks
optimization_runner('tests/test_data/sample_data/portfolio_transition/config_simple.yaml')
optimization_runner('tests/test_data/sample_data/portfolio_transition/config_main.yaml')
optimization_runner('tests/test_data/sample_data/portfolio_transition/config_advanced.yaml')

optimization_runner('tests/test_data/sample_data/product_mix/config_simple.yaml')
optimization_runner('tests/test_data/sample_data/product_mix/config_main.yaml')
optimization_runner('tests/test_data/sample_data/product_mix/config_advanced.yaml')


