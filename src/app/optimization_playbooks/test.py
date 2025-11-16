




    def build_optimization_model(self, processed_data: Dict[str, Any]) -> None:
        """Build MILP model for tax-efficient liquidation."""
        lots = processed_data['lots']
        shares_required = processed_data['shares_required']
        tickers = processed_data['tickers']

        # Decision variables: fraction of each lot to sell (0 to 1)
        for idx, row in lots.iterrows():
            var_name = f"lot_{row['lot_id']}"
            self.lot_variables[row['lot_id']] = self.optimizer.add_variable(
                name=var_name,
                var_type='continuous',
                lb=0.0,
                ub=1.0
            )

            # Binary variable for lot selection (used in lot count constraints)
            bin_var_name = f"lot_sel_{row['lot_id']}"
            self.lot_binary_vars[row['lot_id']] = self.optimizer.add_variable(
                name=bin_var_name,
                var_type='binary'
            )

            # Link binary to continuous: if lot_var > 0, then binary = 1
            # lot_var <= binary (ensures binary is 1 when selling)
            self.optimizer.add_constraint(
                name=f"link_binary_{row['lot_id']}",
                expression=self.lot_variables[row['lot_id']] - self.lot_binary_vars[row['lot_id']],
                constraint_type='leq',
                rhs=0.0
            )

        # Calculate component expressions
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

        # Build objective based on config
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

        # ========================================
        # VALIDATE SHARE REQUIREMENT CONSTRAINT (MANDATORY)
        # ========================================

        constraints_config = self.config.get('constraints', [])

        # Check if user specified the mandatory share requirement constraint
        has_share_constraint = any(
            c.get('type') == 'range_shares_required'
            for c in constraints_config
        )

        if not has_share_constraint:
            raise ValueError(
                "Share requirement constraint is MANDATORY.\n"
                "Add 'range_shares_required' to your config.yaml under 'constraints' section.\n"
                "Example:\n"
                "  - type: range_shares_required\n"
                "    apply_to: all\n"
                "    min_percentage: 1.0  # For full liquidation\n"
                "    max_percentage: 1.0\n"
            )

        for constraint in constraints_config:
            constraint_type = constraint.get('type')

            # ========================================
            # SHARE REQUIREMENT CONSTRAINT (MANDATORY)
            # ========================================

            if constraint_type == 'range_shares_required':
                # Range of shares (min and max percentage)
                apply_to = constraint.get('apply_to', 'all')
                min_pct = constraint.get('min_percentage', 1.0)
                max_pct = constraint.get('max_percentage', 1.0)

                if apply_to == 'all':
                    target_tickers = tickers
                else:
                    target_tickers = apply_to if isinstance(apply_to, list) else [apply_to]

                print(f"\n🔍 DEBUG: range_shares_required constraint")
                print(f"   apply_to: {apply_to}")
                print(f"   target_tickers: {target_tickers}")
                print(f"   min_percentage: {min_pct}, max_percentage: {max_pct}")

                for ticker in target_tickers:
                    if ticker not in tickers:
                        print(f"   ⚠️  Skipping {ticker} - not in available tickers: {tickers}")
                        continue

                    ticker_lots = lots[lots['ticker'] == ticker]
                    required_shares = shares_required[ticker]

                    print(f"   ✓ Adding constraint for {ticker}:")
                    print(f"      Total shares available: {required_shares}")
                    print(
                        f"      Must sell between: {required_shares * min_pct:.2f} - {required_shares * max_pct:.2f} shares")

                    constraint_expr = sum(
                        self.lot_variables[row['lot_id']] * row['shares_purchased']
                        for _, row in ticker_lots.iterrows()
                    )
                    # Minimum
                    self.optimizer.add_constraint(
                        name=f"min_shares_{ticker}",
                        expression=constraint_expr,
                        constraint_type='geq',
                        rhs=shares_required[ticker] * min_pct
                    )
                    # Maximum
                    self.optimizer.add_constraint(
                        name=f"max_shares_{ticker}",
                        expression=constraint_expr,
                        constraint_type='leq',
                        rhs=shares_required[ticker] * max_pct
                    )

            # ========================================
            # CAPITAL GAINS CONSTRAINTS
            # ========================================

            elif constraint_type == 'max_total_capital_gains':
                # Maximum total capital gains
                max_gain = constraint.get('max_gain')
                total_gains_expr = st_gains_expr + st_losses_expr + lt_gains_expr + lt_losses_expr
                self.optimizer.add_constraint(
                    name='max_total_capital_gains',
                    expression=total_gains_expr,
                    constraint_type='leq',
                    rhs=max_gain
                )

            elif constraint_type == 'max_short_term_gains':
                # Maximum short-term capital gains
                max_gain = constraint.get('max_gain')
                st_net_expr = st_gains_expr + st_losses_expr
                self.optimizer.add_constraint(
                    name='max_short_term_gains',
                    expression=st_net_expr,
                    constraint_type='leq',
                    rhs=max_gain
                )

            elif constraint_type == 'max_long_term_gains':
                # Maximum long-term capital gains
                max_gain = constraint.get('max_gain')
                lt_net_expr = lt_gains_expr + lt_losses_expr
                self.optimizer.add_constraint(
                    name='max_long_term_gains',
                    expression=lt_net_expr,
                    constraint_type='leq',
                    rhs=max_gain
                )

            elif constraint_type == 'min_capital_loss_harvesting':
                # Minimum capital loss (negative value means loss)
                min_loss = constraint.get('min_loss')
                total_gains_expr = st_gains_expr + st_losses_expr + lt_gains_expr + lt_losses_expr
                self.optimizer.add_constraint(
                    name='min_capital_loss_harvesting',
                    expression=total_gains_expr,
                    constraint_type='leq',
                    rhs=min_loss
                )

            elif constraint_type == 'max_st_lt_gain_ratio':
                # Short-term gains should be at most X times long-term gains
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
                # Minimum total proceeds from sales
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
                # Maximum total proceeds from sales
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