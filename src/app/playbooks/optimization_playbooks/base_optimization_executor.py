"""
Generic Playbook for Config-Driven Optimization
Handles all optimization workflows through YAML configuration.
"""

from abc import ABC
from typing import Dict, List, Optional, Any, Tuple
import time
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json

from app.models.optimizers.base_optimizer import BaseOptimizer
from app.models.optimizers.milp import MILPOptimizer


class GenericPlaybook(ABC):
    """Generic config-driven optimization playbook."""

    def __init__(self, config: Dict[str, Any], optimizer: Optional[BaseOptimizer] = None):
        """
        Initialize playbook.

        Args:
            config: Configuration dictionary from YAML
            optimizer: Pre-configured optimizer instance (optional)
        """
        self.config = config
        self.optimizer = optimizer
        self.result: Optional[Dict[str, Any]] = None
        self.data: Dict[str, pd.DataFrame] = {}
        self.variables: Dict[str, Any] = {}
        self.output_dir: Optional[Path] = None

    def execute(self) -> Dict[str, Any]:
        """Execute the complete optimization workflow."""
        start_time = time.time()

        try:
            print("\n" + "=" * 70)
            print(f"MODEL: {self.config.get('model_name', 'Optimization')}")
            print(f"TYPE: {self.config.get('playbook_type', 'generic')}")
            print("=" * 70)

            # Setup output directory
            self._setup_output_directory()
            print(f"\n📁 Output Directory: {self.output_dir}")

            # Step 1: Load data
            print("\n⏳ Loading data...")
            self.data = self._load_data()

            # Step 2: Initialize optimizer
            if self.optimizer is None:
                self.optimizer = self._create_optimizer()
            print(f"⏳ Optimizer: {self.optimizer.solver}")

            # Step 3: Create decision variables
            print("⏳ Creating decision variables...")
            self._create_variables()

            # Step 4: Set objective
            print("⏳ Setting objective function...")
            self._set_objective()

            # Step 5: Add constraints
            print("⏳ Adding constraints...")
            self._add_constraints()

            # Solve
            print("\n🔄 Solving optimization problem...")
            opt_result = self.optimizer.solve()

            # Extract solution (playbook-specific)
            solution = self._extract_solution(opt_result)

            # Save outputs (playbook-specific)
            if opt_result['status'] == 'optimal':
                print("\n💾 Saving results...")
                self._save_outputs(solution, opt_result)

            execution_time = time.time() - start_time

            self.result = {
                'status': 'success' if opt_result['status'] == 'optimal' else 'partial',
                'optimization_result': opt_result,
                'solution': solution,
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat(),
                'output_directory': str(self.output_dir)
            }

            # Summary
            print("\n" + "=" * 70)
            print("OPTIMIZATION COMPLETE")
            print("=" * 70)
            print(f"Status: {opt_result['status'].upper()}")
            print(f"Objective Value: {opt_result.get('objective_value', 'N/A')}")
            print(f"Solver Time: {opt_result.get('solver_time', 0):.2f}s")
            print(f"Total Time: {execution_time:.2f}s")
            print(f"Results: {self.output_dir}")
            print("=" * 70 + "\n")

            return self.result

        except Exception as e:
            print("\n" + "=" * 70)
            print("❌ OPTIMIZATION FAILED")
            print("=" * 70)
            print(f"Error: {str(e)}")
            print("=" * 70 + "\n")

            import traceback
            traceback.print_exc()

            return {
                'status': 'failed',
                'error': str(e),
                'execution_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }

    def _setup_output_directory(self) -> None:
        """Setup output directory for saving results."""
        base_output = self.config.get('output', {}).get('directory', 'outputs')
        model_id = self.config.get('model_id')

        if model_id:
            self.output_dir = Path(base_output) / model_id
        else:
            model_name = self.config.get('model_name', 'optimization')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(base_output) / model_name / timestamp

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data files specified in config."""
        data = {}
        data_config = self.config.get('data', {})
        base_path = Path(self.config.get('base_path', '.'))

        for table_name, file_path in data_config.items():
            full_path = base_path / file_path
            data[table_name] = pd.read_csv(full_path)
            print(f"  ✓ {table_name}: {len(data[table_name])} rows")

        return data

    def _create_optimizer(self) -> BaseOptimizer:
        """Create optimizer instance based on config."""
        solver_config = self.config.get('solver', {})
        solver_type = solver_config.get('type', 'APOPT')

        optimizer_params = {
            'solver': solver_type,
            'remote': solver_config.get('remote', False),
            'time_limit': solver_config.get('time_limit'),
            'max_iter': solver_config.get('max_iter'),
            'mip_gap': solver_config.get('mip_gap')
        }

        # Remove None values
        optimizer_params = {k: v for k, v in optimizer_params.items() if v is not None}

        return MILPOptimizer(**optimizer_params)

    def _create_variables(self) -> None:
        """Create decision variables from config."""
        var_defs = self.config.get('variables', [])

        total_vars = 0
        for var_def in var_defs:
            var_name = var_def['name']
            index_spec = var_def['index']  # Format: "table.column"
            var_type = var_def['type']
            lb = var_def.get('lb')
            ub = var_def.get('ub')

            # Parse index specification
            table_name, column_name = index_spec.split('.')
            df = self.data[table_name]

            self.variables[var_name] = {}

            for idx, row in df.iterrows():
                index_value = row[column_name]
                full_var_name = f"{var_name}_{index_value}"

                var = self.optimizer.add_variable(
                    name=full_var_name,
                    var_type=var_type,
                    lb=lb,
                    ub=ub
                )

                self.variables[var_name][index_value] = var
                total_vars += 1

        print(f"  ✓ {total_vars} variables created")

    def _set_objective(self) -> None:
        """Set objective function from config."""
        obj_config = self.config.get('objective', {})
        sense = obj_config.get('sense', 'minimize')
        expression = obj_config.get('expression')

        local_vars = {
            'data': self.data,
            'vars': self.variables,
            'sum': sum,
            'min': min,
            'max': max,
            'abs': abs,
            'pd': pd,
            'np': np
        }

        obj_expr = eval(expression, local_vars)
        self.optimizer.set_objective(obj_expr, sense=sense)
        print(f"  ✓ Objective: {sense}")

    def _add_constraints(self) -> None:
        """Add constraints from config."""
        constraints = self.config.get('constraints', [])

        constraint_count = 0
        for idx, constraint in enumerate(constraints):
            expression = constraint['expression']

            local_vars = {
                'data': self.data,
                'vars': self.variables,
                'sum': sum,
                'min': min,
                'max': max,
                'pd': pd,
                'np': np
            }

            # Evaluate expression to get GEKKO constraint object(s)
            result = eval(expression, local_vars)

            # Add constraint(s) through optimizer interface
            if isinstance(result, list):
                for constraint_obj in result:
                    self.optimizer.add_constraint(constraint_obj)
                constraint_count += len(result)
            else:
                self.optimizer.add_constraint(result)
                constraint_count += 1

        print(f"  ✓ {constraint_count} constraints added")

    def _extract_solution(self, opt_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract solution from optimization result. Override in specific playbooks."""
        if opt_result['status'] != 'optimal':
            return {
                'status': opt_result['status'],
                'message': opt_result.get('message', 'No solution available')
            }

        return {
            'status': 'optimal',
            'objective_value': opt_result['objective_value'],
            'variables': opt_result['variables']
        }

    def _save_outputs(self, solution: Dict[str, Any], opt_result: Dict[str, Any]) -> None:
        """Save outputs to files. Override in specific playbooks for custom outputs."""
        pass

    def reset(self) -> None:
        """Reset playbook to initial state."""
        self.result = None
        self.data = {}
        self.variables = {}
        if self.optimizer:
            self.optimizer.reset()