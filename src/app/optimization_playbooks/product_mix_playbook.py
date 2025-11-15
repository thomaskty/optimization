"""
Product Mix Optimization Playbook
Simple LP example: Maximize profit subject to resource constraints.
"""

from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
from pathlib import Path
from datetime import datetime

from app.optimization_playbooks.base_playbook import BasePlaybook
from app.models.optimizers.milp import MILPOptimizer


class ProductMixPlaybook(BasePlaybook):
    """
    Product Mix Optimization - Classic LP Problem.

    Problem:
    - Factory produces multiple products
    - Each product uses different resources (labor, materials, machine time)
    - Each product has different profit margin
    - Limited resources available

    Decision Variables: Quantity to produce of each product (continuous)
    Objective: Maximize total profit OR minimize total cost

    Constraints:
    - REQUIRED (hardcoded): Resource capacity limits
    - OPTIONAL (from config): Demand limits, minimum production, ratios
    """

    def __init__(self, config: Dict[str, Any], optimizer: Optional[MILPOptimizer] = None):
        super().__init__(config, optimizer)

        # Data structures
        self.products_df: Optional[pd.DataFrame] = None
        self.resources_df: Optional[pd.DataFrame] = None
        self.production_vars: Dict[str, Any] = {}

    def load_data(self) -> Dict[str, Any]:
        """Load product mix data from config."""
        input_data = {}
        datasets = self.config.get('datasets', {})

        # Load products, resources, and resource usage
        if 'products' in datasets:
            input_data['products'] = pd.read_csv(datasets['products'])
        if 'resources' in datasets:
            input_data['resources'] = pd.read_csv(datasets['resources'])
        if 'resource_usage' in datasets:
            input_data['resource_usage'] = pd.read_csv(datasets['resource_usage'])

        return input_data

    def validate_input(self, input_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate input data."""
        errors = []

        # Required: products data
        if 'products' not in input_data:
            errors.append("Missing 'products' DataFrame")
        else:
            products = input_data['products']
            required_cols = ['product_name', 'profit_per_unit']
            missing = [c for c in required_cols if c not in products.columns]
            if missing:
                errors.append(f"Products missing columns: {missing}")

        # Required: resources data
        if 'resources' not in input_data:
            errors.append("Missing 'resources' DataFrame")
        else:
            resources = input_data['resources']
            required_cols = ['resource_name', 'available_capacity']
            missing = [c for c in required_cols if c not in resources.columns]
            if missing:
                errors.append(f"Resources missing columns: {missing}")

        # Required: resource usage data
        if 'resource_usage' not in input_data:
            errors.append("Missing 'resource_usage' DataFrame")
        else:
            usage = input_data['resource_usage']
            required_cols = ['product_name', 'resource_name', 'usage_per_unit']
            missing = [c for c in required_cols if c not in usage.columns]
            if missing:
                errors.append(f"Resource usage missing columns: {missing}")

        return len(errors) == 0, errors

    def preprocess_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for optimization."""
        products = input_data['products'].copy()
        resources = input_data['resources'].copy()
        usage = input_data['resource_usage'].copy()

        self.products_df = products
        self.resources_df = resources

        return {
            'products': products,
            'resources': resources,
            'resource_usage': usage,
            'num_products': len(products),
            'num_resources': len(resources)
        }

    def build_optimization_model(self, processed_data: Dict[str, Any]) -> None:
        """Build LP model for product mix optimization."""
        products = processed_data['products']
        resources = processed_data['resources']
        usage = processed_data['resource_usage']

        # Decision variables: Quantity to produce of each product
        for _, row in products.iterrows():
            product_name = row['product_name']
            var_name = f"qty_{product_name}"
            self.production_vars[product_name] = self.optimizer.add_variable(
                name=var_name,
                var_type='continuous',
                lb=0,  # Can't produce negative quantities
                ub=None  # No upper limit unless constrained
            )

        # Build objective based on config
        objective_config = self.config.get('objective', {})
        objective_function = objective_config.get('function', 'maximize_profit')

        if objective_function == 'maximize_profit':
            # Maximize total profit
            profit_expr = sum(
                self.production_vars[row['product_name']] * row['profit_per_unit']
                for _, row in products.iterrows()
            )
            self.optimizer.set_objective(profit_expr, sense='maximize')

        elif objective_function == 'minimize_cost':
            # Minimize total cost (if cost_per_unit column exists)
            if 'cost_per_unit' not in products.columns:
                raise ValueError("minimize_cost requires 'cost_per_unit' column in products data")
            cost_expr = sum(
                self.production_vars[row['product_name']] * row['cost_per_unit']
                for _, row in products.iterrows()
            )
            self.optimizer.set_objective(cost_expr, sense='minimize')

        elif objective_function == 'maximize_revenue':
            # Maximize total revenue (if price_per_unit exists)
            if 'price_per_unit' not in products.columns:
                raise ValueError("maximize_revenue requires 'price_per_unit' column")
            revenue_expr = sum(
                self.production_vars[row['product_name']] * row['price_per_unit']
                for _, row in products.iterrows()
            )
            self.optimizer.set_objective(revenue_expr, sense='maximize')

        else:
            raise ValueError(f"Unknown objective function: {objective_function}")

        # ========================================
        # REQUIRED CONSTRAINTS (Hardcoded - Business Logic)
        # ========================================

        # Constraint: Resource capacity limits (ALWAYS REQUIRED)
        for _, resource_row in resources.iterrows():
            resource_name = resource_row['resource_name']
            available = resource_row['available_capacity']

            # Get all products that use this resource
            resource_usage = usage[usage['resource_name'] == resource_name]

            if len(resource_usage) > 0:
                usage_expr = sum(
                    self.production_vars[row['product_name']] * row['usage_per_unit']
                    for _, row in resource_usage.iterrows()
                )

                self.optimizer.add_constraint(
                    name=f"resource_limit_{resource_name}",
                    expression=usage_expr,
                    constraint_type='leq',
                    rhs=available
                )

        # ========================================
        # OPTIONAL CONSTRAINTS (From Config - User Preferences)
        # ========================================

        constraints_config = self.config.get('constraints', [])

        for constraint in constraints_config:
            constraint_type = constraint.get('type')

            if constraint_type == 'max_production':
                # Maximum production limit per product
                product = constraint.get('product')
                max_qty = constraint.get('max_quantity')

                if product in self.production_vars:
                    self.optimizer.add_constraint(
                        name=f"max_prod_{product}",
                        expression=self.production_vars[product],
                        constraint_type='leq',
                        rhs=max_qty
                    )

            elif constraint_type == 'min_production':
                # Minimum production requirement
                product = constraint.get('product')
                min_qty = constraint.get('min_quantity')

                if product in self.production_vars:
                    self.optimizer.add_constraint(
                        name=f"min_prod_{product}",
                        expression=self.production_vars[product],
                        constraint_type='geq',
                        rhs=min_qty
                    )

            elif constraint_type == 'production_ratio':
                # Ratio between two products (e.g., produce 2x as much A as B)
                product_a = constraint.get('product_a')
                product_b = constraint.get('product_b')
                ratio = constraint.get('ratio')  # A = ratio * B

                if product_a in self.production_vars and product_b in self.production_vars:
                    # A - ratio * B = 0
                    ratio_expr = (self.production_vars[product_a] -
                                  ratio * self.production_vars[product_b])
                    self.optimizer.add_constraint(
                        name=f"ratio_{product_a}_{product_b}",
                        expression=ratio_expr,
                        constraint_type='eq',
                        rhs=0
                    )

            elif constraint_type == 'total_production_limit':
                # Total production across all products
                max_total = constraint.get('max_total_quantity')
                total_expr = sum(self.production_vars.values())
                self.optimizer.add_constraint(
                    name='total_production_limit',
                    expression=total_expr,
                    constraint_type='leq',
                    rhs=max_total
                )

            elif constraint_type == 'minimum_total_production':
                # Minimum total production requirement
                min_total = constraint.get('min_total_quantity')
                total_expr = sum(self.production_vars.values())
                self.optimizer.add_constraint(
                    name='min_total_production',
                    expression=total_expr,
                    constraint_type='geq',
                    rhs=min_total
                )

    def extract_solution(self, opt_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract production plan from solution."""
        if opt_result['variables'] is None:
            return {
                'status': 'error',
                'production_plan': [],
                'summary': {},
                'error': 'No solution available'
            }

        # Extract production quantities
        production_plan = []

        for product_name in self.production_vars.keys():
            var_name = f"qty_{product_name}"
            quantity = opt_result['variables'].get(var_name, 0.0)
            product_info = self.products_df[
                self.products_df['product_name'] == product_name
                ].iloc[0]

            profit = quantity * product_info['profit_per_unit']

            production_plan.append({
                'product': product_name,
                'quantity': float(quantity),
                'profit_per_unit': float(product_info['profit_per_unit']),
                'total_profit': float(profit)
            })

        # Calculate summary
        total_quantity = sum(p['quantity'] for p in production_plan)
        total_profit = sum(p['total_profit'] for p in production_plan)

        return {
            'status': opt_result['status'],
            'production_plan': production_plan,
            'summary': {
                'total_quantity_produced': float(total_quantity),
                'total_profit': float(total_profit),
                'objective_value': float(opt_result['objective_value']) if opt_result['objective_value'] else 0
            }
        }

    def generate_output(self, solution: Dict[str, Any]) -> Dict[str, Any]:
        """Generate output files."""
        if 'error' in solution:
            return solution

        production_df = pd.DataFrame(solution['production_plan'])

        output_dir = self.config.get('output', {}).get('directory')
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save production plan
            production_df.to_csv(output_path / 'production_plan.csv', index=False)

            # Save summary
            summary_df = pd.DataFrame([solution['summary']])
            summary_df.to_csv(output_path / 'summary.csv', index=False)

            # Save JSON
            import json
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(output_path / f'result_{timestamp}.json', 'w') as f:
                json.dump(solution, f, indent=2, default=str)

        return solution