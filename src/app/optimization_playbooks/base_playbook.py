"""
Base Playbook for Optimization Workflows
Abstract interface for creating reusable optimization playbooks.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
import time
from datetime import datetime

from app.models.optimizers.base_optimizer import BaseOptimizer


class BasePlaybook(ABC):
    """Abstract base class for optimization playbooks."""

    def __init__(self, config: Dict[str, Any], optimizer: Optional[BaseOptimizer] = None):
        """
        Initialize playbook.

        Args:
            config: Configuration dictionary
            optimizer: Pre-configured optimizer instance (optional)
        """
        self.config = config
        self.optimizer = optimizer
        self.result: Optional[Dict[str, Any]] = None

    @abstractmethod
    def load_data(self) -> Dict[str, Any]:
        """Load input data based on config. Each playbook implements its own data loading."""
        pass

    @abstractmethod
    def validate_input(self, input_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate input data. Returns (is_valid, errors)"""
        pass

    @abstractmethod
    def preprocess_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess and transform input data."""
        pass

    @abstractmethod
    def build_optimization_model(self, processed_data: Dict[str, Any]) -> None:
        """Build the optimization model."""
        pass

    @abstractmethod
    def extract_solution(self, opt_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and format the optimization solution."""
        pass

    @abstractmethod
    def generate_output(self, solution: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final output from solution."""
        pass

    def execute(self) -> Dict[str, Any]:
        """Execute the complete playbook workflow."""
        start_time = time.time()
        try:
            # Load data (playbook-specific)
            input_data = self.load_data()

            # Validate input
            is_valid, errors = self.validate_input(input_data)
            if not is_valid:
                return {
                    'status': 'failed',
                    'errors': errors,
                    'execution_time': time.time() - start_time
                }

            # Preprocess data
            processed_data = self.preprocess_data(input_data)

            # Initialize optimizer if not provided
            if self.optimizer is None:
                self.optimizer = self._create_optimizer()

            # Build model
            self.build_optimization_model(processed_data)

            # Solve
            opt_result = self.optimizer.solve()

            # Extract solution
            solution = self.extract_solution(opt_result)

            # Generate output
            output = self.generate_output(solution)

            # Build result
            self.result = {
                'status': 'success' if opt_result['status'] == 'optimal' else 'partial',
                'optimization_result': opt_result,
                'output': output,
                'execution_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }

            return self.result

        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'execution_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }

    def _create_optimizer(self) -> BaseOptimizer:
        """Create optimizer instance based on config."""
        from app.models.optimizers.milp import MILPOptimizer

        model_type = self.config.get('model_type', 'milp').lower()
        optimizer_params = self.config.get('optimizer_params', {})

        if model_type == 'milp':
            return MILPOptimizer(**optimizer_params)
        else:
            raise ValueError(f"Unknown optimizer type: {model_type}")

    def reset(self) -> None:
        """Reset playbook to initial state."""
        self.result = None
        if self.optimizer:
            self.optimizer.reset()