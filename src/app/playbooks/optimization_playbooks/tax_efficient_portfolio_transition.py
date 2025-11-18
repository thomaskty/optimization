"""
Tax-Efficient Portfolio Transition Playbook
Extends GenericPlaybook with specific output formatting for tax optimization.
"""

from typing import Dict, Any
import pandas as pd
import json
from datetime import datetime

from app.playbooks.optimization_playbooks.base_playbook import GenericPlaybook


class TaxEfficientPortfolioTransition(GenericPlaybook):
    """
    Tax-Efficient Portfolio Transition Playbook.
    Uses generic config-driven framework with custom solution extraction and output.
    """

    def _extract_solution(self, opt_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and format solution with tax-specific details."""
        if opt_result['status'] != 'optimal':
            return {
                'status': 'error',
                'message': f"Optimization failed: {opt_result.get('message', 'No solution')}"
            }

        purchase_history = self.data['purchase_history']
        sell_decisions = []

        for idx, row in purchase_history.iterrows():
            lot_id = row['lot_id']
            var_name = f"lot_{lot_id}"
            sell_fraction = opt_result['variables'].get(var_name, 0.0)

            sell_decisions.append({
                'ticker': row['ticker'],
                'lot_id': int(lot_id),
                'acquisition_date': str(row['acquisition_date']),
                'shares_purchased': float(row['shares_purchased']),
                'purchase_price': float(row['purchase_price']),
                'current_price': float(row['current_price']),
                'sell_fraction': float(sell_fraction),
                'shares_to_sell': float(sell_fraction * row['shares_purchased']),
                'capital_gain': float(sell_fraction * row['total_capital_gain']),
                'is_long_term': bool(row['is_long_term']),
                'holding_days': int(row['holding_days']),
                'proceeds': float(sell_fraction * row['proceeds'])
            })

        sold_lots = [s for s in sell_decisions if s['sell_fraction'] > 0.001]

        total_st_gains = sum(s['capital_gain'] for s in sold_lots if not s['is_long_term'] and s['capital_gain'] > 0)
        total_st_losses = sum(s['capital_gain'] for s in sold_lots if not s['is_long_term'] and s['capital_gain'] < 0)
        total_lt_gains = sum(s['capital_gain'] for s in sold_lots if s['is_long_term'] and s['capital_gain'] > 0)
        total_lt_losses = sum(s['capital_gain'] for s in sold_lots if s['is_long_term'] and s['capital_gain'] < 0)

        net_st_gain = total_st_gains + total_st_losses
        net_lt_gain = total_lt_gains + total_lt_losses
        total_net_gain = net_st_gain + net_lt_gain

        tax_params = self.config.get('objective', {}).get('parameters', {})
        st_rate = tax_params.get('short_term_rate', 0.37)
        lt_rate = tax_params.get('long_term_rate', 0.20)

        estimated_tax = max(0, net_st_gain) * st_rate + max(0, net_lt_gain) * lt_rate
        total_proceeds = sum(s['proceeds'] for s in sold_lots)

        return {
            'status': 'optimal',
            'objective_value': opt_result['objective_value'],
            'sell_decisions': sell_decisions,
            'sold_lots': sold_lots,
            'summary': {
                'total_short_term_gains': float(total_st_gains),
                'total_short_term_losses': float(total_st_losses),
                'total_long_term_gains': float(total_lt_gains),
                'total_long_term_losses': float(total_lt_losses),
                'net_short_term_gain': float(net_st_gain),
                'net_long_term_gain': float(net_lt_gain),
                'total_net_capital_gain': float(total_net_gain),
                'estimated_tax': float(estimated_tax),
                'total_proceeds': float(total_proceeds),
                'lots_sold_count': len(sold_lots),
                'short_term_tax_rate': float(st_rate),
                'long_term_tax_rate': float(lt_rate)
            }
        }

    def _save_outputs(self, solution: Dict[str, Any], opt_result: Dict[str, Any]) -> None:
        """Save tax-efficient portfolio outputs."""
        if solution['status'] != 'optimal':
            return

        # Save sell_decisions.csv
        sell_decisions_df = pd.DataFrame(solution['sell_decisions'])
        sell_decisions_df.to_csv(self.output_dir / 'sell_decisions.csv', index=False)

        # Save lots_sold.csv
        lots_sold_df = pd.DataFrame(solution['sold_lots'])
        if len(lots_sold_df) > 0:
            lots_sold_df.to_csv(self.output_dir / 'lots_sold.csv', index=False)

        # Save ticker_summary.csv
        if len(lots_sold_df) > 0:
            ticker_summary = lots_sold_df.groupby('ticker').agg({
                'shares_to_sell': 'sum',
                'capital_gain': 'sum',
                'proceeds': 'sum'
            }).reset_index()
            ticker_summary.columns = ['ticker', 'shares_sold', 'capital_gain', 'proceeds']
            ticker_summary.to_csv(self.output_dir / 'ticker_summary.csv', index=False)

        # Save tax_summary.csv
        tax_summary_df = pd.DataFrame([solution['summary']])
        tax_summary_df.to_csv(self.output_dir / 'tax_summary.csv', index=False)

        # Save optimization_result.json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f'optimization_result_{timestamp}.json'

        result_data = {
            'status': solution['status'],
            'objective_value': solution['objective_value'],
            'sell_decisions': solution['sell_decisions'],
            'summary': solution['summary'],
            'solver_info': {
                'solver_time': opt_result.get('solver_time'),
                'status': opt_result.get('status')
            }
        }

        with open(self.output_dir / result_filename, 'w') as f:
            json.dump(result_data, f, indent=2, default=str)

        print(f"  ✓ sell_decisions.csv")
        print(f"  ✓ lots_sold.csv")
        print(f"  ✓ ticker_summary.csv")
        print(f"  ✓ tax_summary.csv")
        print(f"  ✓ {result_filename}")