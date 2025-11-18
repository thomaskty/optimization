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

from app.utils.typings import OPTIMIZER_TYPES


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

            # Step 2: Apply calculated fields
            print("⏳ Processing calculated fields...")
            self._apply_calculated_fields()

            # Step 3: Initialize optimizer
            if self.optimizer is None:
                self.optimizer = self._create_optimizer()
            print(f"⏳ Optimizer: {self.optimizer.solver}")

            # Step 4: Create decision variables
            print("⏳ Creating decision variables...")
            self._create_variables()

            # Step 5: Set objective
            print("⏳ Setting objective function...")
            self._set_objective()

            # Step 6: Add constraints
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
        files = data_config.get('files', {})

        base_path = Path(self.config.get('base_path', '.'))

        for table_name, file_path in files.items():
            full_path = base_path / file_path
            data[table_name] = pd.read_csv(full_path)
            print(f"  ✓ {table_name}: {len(data[table_name])} rows")

        return data

    def _apply_calculated_fields(self) -> None:
        """Apply calculated fields from config."""
        calc_fields = self.config.get('calculated_fields', {})

        for table_name, fields in calc_fields.items():
            if table_name not in self.data:
                continue

            df = self.data[table_name]

            for field_def in fields:
                for field_name, expression in field_def.items():
                    try:
                        if field_name.startswith('_merge'):
                            local_vars = {
                                'data': self.data,
                                'pd': pd,
                                'np': np
                            }
                            self.data[table_name] = eval(expression, local_vars)
                            df = self.data[table_name]
                        else:
                            local_vars = {
                                'pd': pd,
                                'np': np,
                                'data': self.data
                            }
                            for col in df.columns:
                                local_vars[col] = df[col]

                            df[field_name] = eval(expression, local_vars)

                    except Exception as e:
                        raise Exception(f"Error in {table_name}.{field_name}: {str(e)}")

            self.data[table_name] = df

        total_fields = sum(len(fields) for fields in calc_fields.values())
        print(f"  ✓ {total_fields} calculated fields applied")

    def _create_optimizer(self) -> BaseOptimizer:
        """Create optimizer instance based on config."""
        model_type = self.config.get('model_type', None).lower()
        optimizer_params = self.config.get('optimizer_params', {})

        return OPTIMIZER_TYPES.get(model_type)(**optimizer_params)

    def _create_variables(self) -> None:
        """Create decision variables from config."""
        var_defs = self.config.get('decision_variables', [])

        total_vars = 0
        for var_def in var_defs:
            var_name = var_def['name']
            template = var_def['template']
            index_from = var_def['index_from']
            index_col = var_def['index_column']
            var_type = var_def['type']
            bounds = var_def.get('bounds', [None, None])

            df = self.data[index_from]
            self.variables[var_name] = {}

            for idx, row in df.iterrows():
                index_value = row[index_col]
                full_var_name = template.format(**{index_col: index_value})

                var = self.optimizer.add_variable(
                    name=full_var_name,
                    var_type=var_type,
                    lb=bounds[0] if len(bounds) > 0 else None,
                    ub=bounds[1] if len(bounds) > 1 else None
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
        for constraint in constraints:
            constraint_type_def = constraint['type']

            if constraint_type_def == 'for_each':
                count = self._add_for_each_constraint(constraint)
                constraint_count += count
            elif constraint_type_def == 'for_each_group':
                count = self._add_for_each_group_constraint(constraint)
                constraint_count += count
            elif constraint_type_def == 'single':
                self._add_single_constraint(constraint)
                constraint_count += 1

        print(f"  ✓ {constraint_count} constraints added")

    def _add_for_each_constraint(self, constraint: Dict[str, Any]) -> int:
        """Add constraint for each row in a table."""
        iterate_over = constraint['iterate_over']
        expression = constraint['expression']
        constraint_type = constraint['constraint_type']
        rhs = constraint.get('rhs', 0)

        df = self.data[iterate_over]
        count = 0

        for idx, row in df.iterrows():
            local_vars = {
                'row': row,
                'vars': self.variables,
                'data': self.data
            }

            expr = eval(expression, local_vars)

            self.optimizer.add_constraint(
                name=f"{constraint['name']}_{idx}",
                expression=expr,
                constraint_type=constraint_type,
                rhs=rhs
            )
            count += 1

        return count

    def _add_for_each_group_constraint(self, constraint: Dict[str, Any]) -> int:
        """Add constraint for each group in a table."""
        iterate_over = constraint['iterate_over']
        group_by = constraint['group_by']
        expression = constraint['expression']
        constraint_type = constraint['constraint_type']
        rhs_expression = constraint.get('rhs_expression')
        rhs = constraint.get('rhs', 0)

        df = self.data[iterate_over]
        grouped = df.groupby(group_by)
        count = 0

        for group_name, group_data in grouped:
            local_vars = {
                'group_data': group_data,
                'group_name': group_name,
                'vars': self.variables,
                'data': self.data
            }

            expr = eval(expression, local_vars)

            if rhs_expression:
                rhs_value = eval(rhs_expression, local_vars)
            else:
                rhs_value = rhs

            self.optimizer.add_constraint(
                name=f"{constraint['name']}_{group_name}",
                expression=expr,
                constraint_type=constraint_type,
                rhs=rhs_value
            )
            count += 1

        return count

    def _add_single_constraint(self, constraint: Dict[str, Any]) -> None:
        """Add a single constraint."""
        expression = constraint['expression']
        constraint_type = constraint['constraint_type']
        rhs = constraint.get('rhs', 0)

        local_vars = {
            'data': self.data,
            'vars': self.variables,
            'sum': sum,
            'min': min,
            'max': max
        }

        expr = eval(expression, local_vars)

        self.optimizer.add_constraint(
            name=constraint['name'],
            expression=expr,
            constraint_type=constraint_type,
            rhs=rhs
        )

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