"""
----------------------------------------------------------------------------------
Tax-Efficient Portfolio Transition Playbook
Optimize stock liquidation with short-term/long-term capital gains considerations.
----------------------------------------------------------------------------------
you want to move a client's existing holdings from one platform to another,
but you don't want to blindly sell everything and trigger large capital gains.
The trick is to sell only what's required and buy back
in a way that keeps the portfolio close to the original while staying under a gain limit.
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from pathlib import Path
from datetime import datetime

from app.optimization_playbooks.base_playbook import BasePlaybook
from app.models.optimizers.milp import MILPOptimizer


class TaxEfficientPortfolioTransition(BasePlaybook):
    """
    Tax-Efficient Portfolio Transition Playbook.

    Optimizes stock liquidation considering:
    - Short-term vs long-term capital gains (different tax rates)
    - Capital loss harvesting and offsetting
    - Ordinary income offset limits
    """

    def __init__(self, config: Dict[str, Any], optimizer: Optional[MILPOptimizer] = None):
        super().__init__(config, optimizer)

        # Tax parameters
        tax_params = config.get('tax_parameters', {})
        self.short_term_cg_rate = tax_params.get('short_term_cg_rate', 0.37)
        self.long_term_cg_rate = tax_params.get('long_term_cg_rate', 0.20)
        self.ordinary_income_offset_limit = tax_params.get('ordinary_income_offset_limit', 3000)
        self.long_term_threshold_days = tax_params.get('long_term_threshold_days', 365)

        valuation_date_str = tax_params.get('valuation_date', datetime.now())
        self.valuation_date = pd.to_datetime(valuation_date_str)

        # Data structures
        self.lots: Optional[pd.DataFrame] = None
        self.lot_variables: Dict[int, Any] = {}
        self.lot_binary_vars: Dict[int, Any] = {}

    def load_data(self) -> Dict[str, Any]:
        """Load tax-efficient portfolio transition data from config."""
        input_data = {}
        datasets = self.config.get('datasets', {})

        if 'holdings' in datasets:
            input_data['holdings'] = pd.read_csv(datasets['holdings'])
        if 'acquisitions' in datasets:
            input_data['purchase_history'] = pd.read_csv(datasets['acquisitions'])

        return input_data

    def validate_input(self, input_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate input data."""
        errors = []

        if 'holdings' not in input_data:
            errors.append("Missing 'holdings' DataFrame")
        if 'purchase_history' not in input_data:
            errors.append("Missing 'purchase_history' DataFrame")

        if errors:
            return False, errors

        holdings = input_data['holdings']
        required_cols = ['ticker', 'shares_held', 'current_price', 'total_value']
        missing = [col for col in required_cols if col not in holdings.columns]
        if missing:
            errors.append(f"Holdings missing columns: {missing}")

        purchase_history = input_data['purchase_history']
        required_cols = ['ticker', 'acquisition_date', 'purchase_price', 'shares_purchased']
        missing = [col for col in required_cols if col not in purchase_history.columns]
        if missing:
            errors.append(f"Purchase history missing columns: {missing}")

        # VALIDATE MANDATORY SHARE REQUIREMENT CONSTRAINT
        constraints_config = self.config.get('constraints', [])
        has_share_constraint = any(
            c.get('type') == 'range_shares_required'
            for c in constraints_config
        )

        if not has_share_constraint:
            errors.append(
                "MANDATORY: 'range_shares_required' constraint missing.\n"
                "    Example:\n"
                "      - type: range_shares_required\n"
                "        apply_to: all\n"
                "        min_percentage: 1.0  # Full liquidation\n"
                "        max_percentage: 1.0"
            )

        return len(errors) == 0, errors

    def preprocess_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess data for optimization with tax considerations."""
        holdings = input_data['holdings'].copy()
        purchase_history = input_data['purchase_history'].copy()

        # Convert dates and sort by acquisition date (FIFO)
        purchase_history['acquisition_date'] = pd.to_datetime(purchase_history['acquisition_date'])
        purchase_history = purchase_history.sort_values('acquisition_date').reset_index(drop=True)
        purchase_history['lot_id'] = range(len(purchase_history))

        # Get current prices
        current_prices = holdings.set_index('ticker')['current_price'].to_dict()
        purchase_history['current_price'] = purchase_history['ticker'].map(current_prices)

        # Calculate holding period
        purchase_history['holding_period_days'] = (
                self.valuation_date - purchase_history['acquisition_date']
        ).dt.days

        # Classify as long-term or short-term
        purchase_history['is_long_term'] = purchase_history['holding_period_days'] >= self.long_term_threshold_days

        # Calculate capital gains/losses
        purchase_history['capital_gain_per_share'] = (
                purchase_history['current_price'] - purchase_history['purchase_price']
        )
        purchase_history['total_capital_gain'] = (
                purchase_history['capital_gain_per_share'] * purchase_history['shares_purchased']
        )

        # Classify gains/losses
        purchase_history['is_gain'] = purchase_history['total_capital_gain'] > 0
        purchase_history['is_loss'] = purchase_history['total_capital_gain'] < 0

        # Classify by term and gain/loss
        purchase_history['short_term_gain'] = (
                                                      (~purchase_history['is_long_term']) & purchase_history['is_gain']
                                              ).astype(float) * purchase_history['total_capital_gain']

        purchase_history['short_term_loss'] = (
                                                      (~purchase_history['is_long_term']) & purchase_history['is_loss']
                                              ).astype(float) * purchase_history['total_capital_gain']

        purchase_history['long_term_gain'] = (
                                                     purchase_history['is_long_term'] & purchase_history['is_gain']
                                             ).astype(float) * purchase_history['total_capital_gain']

        purchase_history['long_term_loss'] = (
                                                     purchase_history['is_long_term'] & purchase_history['is_loss']
                                             ).astype(float) * purchase_history['total_capital_gain']

        # Calculate proceeds per lot
        purchase_history['proceeds'] = (
                purchase_history['current_price'] * purchase_history['shares_purchased']
        )

        self.lots = purchase_history
        shares_required = holdings.set_index('ticker')['shares_held'].to_dict()
        tickers = holdings['ticker'].unique().tolist()

        return {
            'lots': purchase_history,
            'holdings': holdings,
            'shares_required': shares_required,
            'tickers': tickers
        }

    def build_optimization_model(self, processed_data: Dict[str, Any]) -> None:
        """Build MILP model for tax-efficient liquidation."""
        lots = processed_data['lots']
        shares_required = processed_data['shares_required']
        tickers = processed_data['tickers']

        # ==========================================================
        # 1) MERGE SHARE REQUIREMENT CONSTRAINTS (DEFAULT + OVERRIDES)
        # ==========================================================
        def merge_range_share_constraints(constraints_cfg, tickers):
            defaults = None
            overrides = []

            # Scan constraints
            for c in constraints_cfg:
                if c.get("type") != "range_shares_required":
                    continue

                apply_to = c.get("apply_to")

                # default block
                if apply_to == "all":
                    defaults = {
                        "min_percentage": c.get("min_percentage", 1.0),
                        "max_percentage": c.get("max_percentage", 1.0),
                    }
                else:
                    overrides.append(c)

            if defaults is None:
                raise ValueError(
                    "Share requirement constraint is MANDATORY.\n"
                    "Add:\n"
                    "  - type: range_shares_required\n"
                    "    apply_to: all\n"
                    "    min_percentage: ...\n"
                    "    max_percentage: ..."
                )

            # Build merged final map
            merged = {t: defaults.copy() for t in tickers}

            for c in overrides:
                apply_to = c["apply_to"]
                if isinstance(apply_to, str):
                    apply_to = [apply_to]

                for t in apply_to:
                    if t not in merged:
                        continue
                    if "min_percentage" in c:
                        merged[t]["min_percentage"] = c["min_percentage"]
                    if "max_percentage" in c:
                        merged[t]["max_percentage"] = c["max_percentage"]

            return merged

        # run merge
        merged_range_constraints = merge_range_share_constraints(
            self.config.get('constraints', []),
            tickers
        )

        # ==========================================================
        # BUILD VARIABLES (unchanged)
        # ==========================================================
        for idx, row in lots.iterrows():
            var_name = f"lot_{row['lot_id']}"
            self.lot_variables[row['lot_id']] = self.optimizer.add_variable(
                name=var_name,
                var_type='continuous',
                lb=0.0,
                ub=1.0
            )

            bin_var_name = f"lot_sel_{row['lot_id']}"
            self.lot_binary_vars[row['lot_id']] = self.optimizer.add_variable(
                name=bin_var_name,
                var_type='binary'
            )

            self.optimizer.add_constraint(
                name=f"link_binary_{row['lot_id']}",
                expression=self.lot_variables[row['lot_id']] - self.lot_binary_vars[row['lot_id']],
                constraint_type='leq',
                rhs=0.0
            )

        # ==========================================================
        # BUILD EXPRESSIONS (unchanged)
        # ==========================================================
        st_gains_expr = sum(
            self.lot_variables[row['lot_id']] * row['short_term_gain']
            for _, row in lots.iterrows()
        )

        st_losses_expr = sum(
            self.lot_variables[row['lot_id']] * row['short_term_loss']
            for _, row in lots.iterrows()
        )

        lt_gains_expr = sum(
            self.lot_variables[row['lot_id']] * row['long_term_gain']
            for _, row in lots.iterrows()
        )

        lt_losses_expr = sum(
            self.lot_variables[row['lot_id']] * row['long_term_loss']
            for _, row in lots.iterrows()
        )

        # ==========================================================
        # OBJECTIVE (unchanged)
        # ==========================================================
        objective_config = self.config.get('objective', {})
        objective_function = objective_config.get('function', 'minimize_tax_liability')

        if objective_function == 'minimize_tax_liability':
            objective_expr = (
                    (st_gains_expr + st_losses_expr) * self.short_term_cg_rate +
                    (lt_gains_expr + lt_losses_expr) * self.long_term_cg_rate
            )
            self.optimizer.set_objective(objective_expr, sense='minimize')

        elif objective_function == 'maximize_capital_losses':
            objective_expr = st_losses_expr + lt_losses_expr
            self.optimizer.set_objective(objective_expr, sense='maximize')

        elif objective_function == 'minimize_short_term_gains':
            objective_expr = st_gains_expr + st_losses_expr
            self.optimizer.set_objective(objective_expr, sense='minimize')

        elif objective_function == 'maximize_long_term_gains':
            objective_expr = lt_gains_expr + lt_losses_expr
            self.optimizer.set_objective(objective_expr, sense='maximize')

        elif objective_function == 'minimize_total_gains':
            objective_expr = st_gains_expr + st_losses_expr + lt_gains_expr + lt_losses_expr
            self.optimizer.set_objective(objective_expr, sense='minimize')

        elif objective_function == 'maximize_total_gains':
            objective_expr = st_gains_expr + st_losses_expr + lt_gains_expr + lt_losses_expr
            self.optimizer.set_objective(objective_expr, sense='maximize')

        elif objective_function == 'weighted_tax_optimization':
            weights = objective_config.get('weights', {})
            st_weight = weights.get('short_term', self.short_term_cg_rate)
            lt_weight = weights.get('long_term', self.long_term_cg_rate)
            objective_expr = (
                    (st_gains_expr + st_losses_expr) * st_weight +
                    (lt_gains_expr + lt_losses_expr) * lt_weight
            )
            self.optimizer.set_objective(objective_expr, sense='minimize')

        else:
            raise ValueError(f"Unknown objective function: {objective_function}")

        # ==========================================================
        # APPLY MERGED SHARE CONSTRAINTS (clean & correct now)
        # ==========================================================
        for ticker in tickers:
            req = merged_range_constraints[ticker]
            min_pct = req["min_percentage"]
            max_pct = req["max_percentage"]

            ticker_lots = lots[lots['ticker'] == ticker]
            required_shares = shares_required[ticker]

            constraint_expr = sum(
                self.lot_variables[row['lot_id']] * row['shares_purchased']
                for _, row in ticker_lots.iterrows()
            )

            # min
            self.optimizer.add_constraint(
                name=f"min_shares_{ticker}",
                expression=constraint_expr,
                constraint_type='geq',
                rhs=required_shares * min_pct
            )

            # max
            self.optimizer.add_constraint(
                name=f"max_shares_{ticker}",
                expression=constraint_expr,
                constraint_type='leq',
                rhs=required_shares * max_pct
            )

        # ==========================================================
        # OTHER CONSTRAINTS (unchanged)
        # ==========================================================
        for constraint in self.config.get('constraints', []):
            constraint_type = constraint.get('type')

            if constraint_type in ("range_shares_required", None):
                continue  # we already handled merged form

            if constraint_type == 'max_total_capital_gains':
                max_gain = constraint.get('max_gain')
                total_gains_expr = st_gains_expr + st_losses_expr + lt_gains_expr + lt_losses_expr
                self.optimizer.add_constraint(
                    name='max_total_capital_gains',
                    expression=total_gains_expr,
                    constraint_type='leq',
                    rhs=max_gain
                )

            elif constraint_type == 'max_short_term_gains':
                max_gain = constraint.get('max_gain')
                st_net_expr = st_gains_expr + st_losses_expr
                self.optimizer.add_constraint(
                    name='max_short_term_gains',
                    expression=st_net_expr,
                    constraint_type='leq',
                    rhs=max_gain
                )

            elif constraint_type == 'max_long_term_gains':
                max_gain = constraint.get('max_gain')
                lt_net_expr = lt_gains_expr + lt_losses_expr
                self.optimizer.add_constraint(
                    name='max_long_term_gains',
                    expression=lt_net_expr,
                    constraint_type='leq',
                    rhs=max_gain
                )

            elif constraint_type == 'min_capital_loss_harvesting':
                min_loss = constraint.get('min_loss')
                total_gains_expr = st_gains_expr + st_losses_expr + lt_gains_expr + lt_losses_expr
                self.optimizer.add_constraint(
                    name='min_capital_loss_harvesting',
                    expression=total_gains_expr,
                    constraint_type='leq',
                    rhs=min_loss
                )

            elif constraint_type == 'max_st_lt_gain_ratio':
                max_ratio = constraint.get('max_ratio')
                st_net_expr = st_gains_expr + st_losses_expr
                lt_net_expr = lt_gains_expr + lt_losses_expr
                ratio_expr = st_net_expr - max_ratio * lt_net_expr
                self.optimizer.add_constraint(
                    name='max_st_lt_gain_ratio',
                    expression=ratio_expr,
                    constraint_type='leq',
                    rhs=0.0
                )

            elif constraint_type == 'min_total_proceeds':
                min_proceeds = constraint.get('min_proceeds')
                proceeds_expr = sum(
                    self.lot_variables[row['lot_id']] * row['proceeds']
                    for _, row in lots.iterrows()
                )
                self.optimizer.add_constraint(
                    name='min_total_proceeds',
                    expression=proceeds_expr,
                    constraint_type='geq',
                    rhs=min_proceeds
                )

            elif constraint_type == 'max_total_proceeds':
                max_proceeds = constraint.get('max_proceeds')
                proceeds_expr = sum(
                    self.lot_variables[row['lot_id']] * row['proceeds']
                    for _, row in lots.iterrows()
                )
                self.optimizer.add_constraint(
                    name='max_total_proceeds',
                    expression=proceeds_expr,
                    constraint_type='leq',
                    rhs=max_proceeds
                )

    def extract_solution(self, opt_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract solution from optimization result."""
        if opt_result['variables'] is None:
            print("\n" + "=" * 60)
            print("OPTIMIZATION FAILED")
            print("=" * 60)
            print(f"Status: {opt_result.get('status')}")
            print(f"Message: {opt_result.get('message')}")
            print(f"Solver Time: {opt_result.get('solver_time')} seconds")
            print("=" * 60 + "\n")
            return {
                'status': 'error',
                'sell_decisions': [],
                'summary': {},
                'error': f"Optimization failed: {opt_result.get('message', 'No solution available')}"
            }

        # Build sell_decisions list
        sell_decisions = []

        for lot_id in self.lot_variables.keys():
            var_name = f"lot_{lot_id}"
            sell_fraction = opt_result['variables'].get(var_name, 0.0)
            lot_info = self.lots[self.lots['lot_id'] == lot_id].iloc[0]

            sell_decisions.append({
                'ticker': lot_info['ticker'],
                'acquisition_date': str(lot_info['acquisition_date'].date()),
                'quantity': int(lot_info['shares_purchased']),
                'quantity_to_sell': float(sell_fraction * lot_info['shares_purchased']),
                'sell': bool(sell_fraction > 0.001),
                'sell_fraction': float(sell_fraction),
                'purchase_price': float(lot_info['purchase_price']),
                'current_price': float(lot_info['current_price']),
                'is_long_term': bool(lot_info['is_long_term']),
                'holding_period_days': int(lot_info['holding_period_days']),
                'capital_gain': float(sell_fraction * lot_info['total_capital_gain']),
                'proceeds': float(sell_fraction * lot_info['proceeds'])
            })

        # Calculate summary statistics
        sold_lots = [lot for lot in sell_decisions if lot['sell']]

        total_short_term_gain = sum(
            lot['capital_gain'] for lot in sold_lots
            if not lot['is_long_term'] and lot['capital_gain'] > 0
        )

        total_short_term_loss = sum(
            lot['capital_gain'] for lot in sold_lots
            if not lot['is_long_term'] and lot['capital_gain'] < 0
        )

        total_long_term_gain = sum(
            lot['capital_gain'] for lot in sold_lots
            if lot['is_long_term'] and lot['capital_gain'] > 0
        )

        total_long_term_loss = sum(
            lot['capital_gain'] for lot in sold_lots
            if lot['is_long_term'] and lot['capital_gain'] < 0
        )

        total_proceeds = sum(lot['proceeds'] for lot in sold_lots)
        lots_sold_count = len(sold_lots)

        # Calculate net amounts
        net_short_term_gain = total_short_term_gain + total_short_term_loss
        net_long_term_gain = total_long_term_gain + total_long_term_loss
        total_net_capital_gain = net_short_term_gain + net_long_term_gain

        # Calculate tax offset
        if total_net_capital_gain < 0:
            tax_offset = min(abs(total_net_capital_gain), self.ordinary_income_offset_limit)
            carryforward_loss = abs(total_net_capital_gain) - tax_offset
            taxable_capital_gain = 0
        else:
            tax_offset = 0
            carryforward_loss = 0
            taxable_capital_gain = total_net_capital_gain

        # Calculate estimated tax
        estimated_tax = (
                max(0, net_short_term_gain) * self.short_term_cg_rate +
                max(0, net_long_term_gain) * self.long_term_cg_rate
        )

        return {
            'status': opt_result['status'],
            'sell_decisions': sell_decisions,
            'summary': {
                'total_short_term_gain': float(total_short_term_gain),
                'total_short_term_loss': float(total_short_term_loss),
                'total_long_term_gain': float(total_long_term_gain),
                'total_long_term_loss': float(total_long_term_loss),
                'net_short_term_gain': float(net_short_term_gain),
                'net_long_term_gain': float(net_long_term_gain),
                'total_net_capital_gain': float(total_net_capital_gain),
                'tax_offset': float(tax_offset),
                'taxable_capital_gain': float(taxable_capital_gain),
                'carryforward_loss': float(carryforward_loss),
                'estimated_tax': float(estimated_tax),
                'total_proceeds': float(total_proceeds),
                'lots_sold_count': int(lots_sold_count),
                'short_term_tax_rate': float(self.short_term_cg_rate),
                'long_term_tax_rate': float(self.long_term_cg_rate)
            }
        }

    def generate_output(self, solution: Dict[str, Any]) -> Dict[str, Any]:
        """Generate formatted output."""
        if 'error' in solution:
            return solution

        summary = solution['summary']
        sell_decisions = solution['sell_decisions']

        # Convert to DataFrame
        sell_decisions_df = pd.DataFrame(sell_decisions)
        lots_sold_df = sell_decisions_df[sell_decisions_df['sell'] == True].copy()

        # Summary by ticker
        if len(lots_sold_df) > 0:
            ticker_summary = lots_sold_df.groupby('ticker').agg({
                'quantity_to_sell': 'sum',
                'capital_gain': 'sum',
                'proceeds': 'sum'
            }).reset_index()
            ticker_summary.columns = ['ticker', 'shares_sold', 'capital_gain', 'proceeds']
        else:
            ticker_summary = pd.DataFrame(columns=['ticker', 'shares_sold', 'capital_gain', 'proceeds'])

        output = {
            'status': solution['status'],
            'sell_decisions': sell_decisions,
            'optimization_summary': summary,
            'ticker_summary': ticker_summary.to_dict('records')
        }

        # Save outputs if directory specified
        output_dir = self.config.get('output', {}).get('directory')
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save CSVs
            sell_decisions_df.to_csv(output_path / 'sell_decisions.csv', index=False)
            if len(lots_sold_df) > 0:
                lots_sold_df.to_csv(output_path / 'lots_sold.csv', index=False)
            ticker_summary.to_csv(output_path / 'ticker_summary.csv', index=False)
            pd.DataFrame([summary]).to_csv(output_path / 'tax_summary.csv', index=False)

            # Save JSON
            import json
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f'optimization_result_{timestamp}.json'
            with open(output_path / result_filename, 'w') as f:
                json.dump({
                    'status': solution['status'],
                    'sell_decisions': sell_decisions,
                    'summary': summary
                }, f, indent=2, default=str)

        return output