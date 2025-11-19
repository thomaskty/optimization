"""
Product Mix Optimization Playbook
Simple manufacturing optimization example.
"""

from typing import Dict, Any
import pandas as pd
import json
from datetime import datetime

from app.playbooks.optimization_playbooks.base_optimization_executor import GenericPlaybook


class ProductMixPlaybook(GenericPlaybook):
    """Product mix optimization with custom output formatting."""

    def _extract_solution(self, opt_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and format solution."""

        print("\n  DEBUG: All variable values from solver:")
        for var_name, var_value in sorted(opt_result['variables'].items()):
            print(f"    {var_name} = {var_value}")

        print(f"\n  DEBUG: Objective from result: {opt_result.get('objective_value')}")

        if opt_result['status'] != 'optimal':
            return {
                'status': 'error',
                'message': f"Optimization failed: {opt_result.get('message', 'No solution')}"
            }

        products = self.data['products']
        production_plan = []

        for _, row in products.iterrows():
            product_id = row['product_id']
            production_qty = opt_result['variables'].get(f"production_{product_id}", 0.0)
            is_produced = opt_result['variables'].get(f"produce_{product_id}", 0.0)

            if production_qty > 0.01:
                production_plan.append({
                    'product_id': int(product_id),
                    'product_name': row['product_name'],
                    'quantity': int(production_qty),
                    'profit_per_unit': float(row['profit_per_unit']),
                    'total_profit': float(production_qty * row['profit_per_unit']),
                    'setup_cost': float(row['setup_cost'] if is_produced > 0.5 else 0),
                    'wood_used': float(production_qty * row['wood_required']),
                    'labor_used': float(production_qty * row['labor_hours']),
                    'machine_used': float(production_qty * row['machine_hours'])
                })

        # Calculate resource utilization
        total_wood = sum(p['wood_used'] for p in production_plan)
        total_labor = sum(p['labor_used'] for p in production_plan)
        total_machine = sum(p['machine_used'] for p in production_plan)
        total_setup_cost = sum(p['setup_cost'] for p in production_plan)
        total_profit = sum(p['total_profit'] for p in production_plan)
        net_profit = total_profit - total_setup_cost

        summary = {
            'total_products_produced': sum(p['quantity'] for p in production_plan),
            'product_types': len(production_plan),
            'total_revenue': float(total_profit),
            'total_setup_cost': float(total_setup_cost),
            'net_profit': float(net_profit),
            'wood_used': float(total_wood),
            'wood_available': 2000.0,
            'wood_utilization': float(total_wood / 2000 * 100),
            'labor_used': float(total_labor),
            'labor_available': 800.0,
            'labor_utilization': float(total_labor / 800 * 100),
            'machine_used': float(total_machine),
            'machine_available': 400.0,
            'machine_utilization': float(total_machine / 400 * 100)
        }

        return {
            'status': 'optimal',
            'objective_value': opt_result['objective_value'],
            'production_plan': production_plan,
            'summary': summary
        }

    def _save_outputs(self, solution: Dict[str, Any], opt_result: Dict[str, Any]) -> None:
        """Save product mix outputs."""
        if solution['status'] != 'optimal':
            return

        # Save production_plan.csv
        production_df = pd.DataFrame(solution['production_plan'])
        production_df.to_csv(self.output_dir / 'production_plan.csv', index=False)

        # Save summary.csv
        summary_df = pd.DataFrame([solution['summary']])
        summary_df.to_csv(self.output_dir / 'summary.csv', index=False)

        # Save optimization_result.json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f'optimization_result_{timestamp}.json'

        result_data = {
            'status': solution['status'],
            'objective_value': solution['objective_value'],
            'production_plan': solution['production_plan'],
            'summary': solution['summary'],
            'solver_info': {
                'solver_time': opt_result.get('solver_time'),
                'status': opt_result.get('status')
            }
        }

        with open(self.output_dir / result_filename, 'w') as f:
            json.dump(result_data, f, indent=2, default=str)

        print(f"  ✓ production_plan.csv")
        print(f"  ✓ summary.csv")
        print(f"  ✓ {result_filename}")