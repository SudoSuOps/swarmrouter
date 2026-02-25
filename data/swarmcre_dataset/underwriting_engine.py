"""
SwarmCRE Dataset Factory — Underwriting Math Engine

Pure deterministic math. Zero randomness. Zero state.
Every method is @staticmethod — trivially testable, provably reproducible.

Rounding rules:
  - Money: round to nearest $1
  - Ratios: round to 2 decimal places
  - Per-SF: round to 2 decimal places
  - Percentages: round to 3 decimal places
"""

import math


class UnderwritingEngine:
    """Deterministic CRE underwriting calculations."""

    # ── Revenue ──────────────────────────────────────────────

    @staticmethod
    def compute_pgi(rent_roll: list[dict]) -> int:
        """Potential Gross Income from rent roll (sum of all annual rents)."""
        return round(sum(t["annual_rent"] for t in rent_roll))

    @staticmethod
    def compute_vacancy_loss(pgi: int, vacancy_rate: float) -> int:
        """Vacancy & credit loss = PGI * vacancy rate."""
        return round(pgi * vacancy_rate)

    @staticmethod
    def compute_egi(pgi: int, vacancy_rate: float, other_income: int = 0) -> int:
        """Effective Gross Income = PGI * (1 - vacancy) + other income."""
        return round(pgi * (1.0 - vacancy_rate) + other_income)

    # ── Expenses ─────────────────────────────────────────────

    @staticmethod
    def compute_opex_line(sf: int, rate_psf: float) -> int:
        """Single expense line = SF * rate per SF."""
        return round(sf * rate_psf)

    @staticmethod
    def compute_management_fee(egi: int, fee_pct: float) -> int:
        """Management fee = EGI * fee percentage."""
        return round(egi * fee_pct)

    @staticmethod
    def compute_total_opex(expense_lines: dict[str, int]) -> int:
        """Sum of all operating expense line items."""
        return sum(expense_lines.values())

    # ── NOI ──────────────────────────────────────────────────

    @staticmethod
    def compute_noi(egi: int, total_opex: int) -> int:
        """Net Operating Income = EGI - Total OpEx."""
        return egi - total_opex

    @staticmethod
    def compute_noi_per_sf(noi: int, sf: int) -> float:
        """NOI per square foot."""
        if sf <= 0:
            return 0.0
        return round(noi / sf, 2)

    # ── Valuation ────────────────────────────────────────────

    @staticmethod
    def compute_value(noi: int, cap_rate: float) -> int:
        """Direct capitalization: Value = NOI / Cap Rate."""
        if cap_rate <= 0:
            return 0
        return round(noi / cap_rate)

    @staticmethod
    def compute_price_per_sf(value: int, sf: int) -> float:
        """Price per square foot."""
        if sf <= 0:
            return 0.0
        return round(value / sf, 2)

    @staticmethod
    def compute_implied_cap(noi: int, value: int) -> float:
        """Implied cap rate = NOI / Value."""
        if value <= 0:
            return 0.0
        return round(noi / value, 4)

    # ── Debt Metrics ─────────────────────────────────────────

    @staticmethod
    def compute_annual_debt_service(
        loan_amount: int, rate: float, amort_years: int, io_years: int = 0
    ) -> int:
        """Compute annual debt service.

        If io_years > 0 (and we're in IO period), DS = loan * rate.
        Otherwise, standard amortizing mortgage payment * 12.
        """
        if loan_amount <= 0 or rate <= 0:
            return 0

        # IO period: interest only
        if amort_years == 0 or io_years > 0:
            return round(loan_amount * rate)

        # Amortizing: PMT formula
        monthly_rate = rate / 12.0
        n_payments = amort_years * 12
        if monthly_rate == 0:
            monthly_payment = loan_amount / n_payments
        else:
            monthly_payment = loan_amount * (
                monthly_rate * (1 + monthly_rate) ** n_payments
            ) / ((1 + monthly_rate) ** n_payments - 1)

        return round(monthly_payment * 12)

    @staticmethod
    def compute_loan_amount_from_ltv(value: int, ltv: float) -> int:
        """Loan amount = Value * LTV."""
        return round(value * ltv)

    @staticmethod
    def compute_dscr(noi: int, annual_debt_service: int) -> float:
        """Debt Service Coverage Ratio = NOI / Annual DS."""
        if annual_debt_service <= 0:
            return 0.0
        return round(noi / annual_debt_service, 2)

    @staticmethod
    def compute_debt_yield(noi: int, loan_amount: int) -> float:
        """Debt Yield = NOI / Loan Amount."""
        if loan_amount <= 0:
            return 0.0
        return round(noi / loan_amount, 4)

    @staticmethod
    def compute_ltv(loan_amount: int, value: int) -> float:
        """Loan-to-Value = Loan / Value."""
        if value <= 0:
            return 0.0
        return round(loan_amount / value, 4)

    @staticmethod
    def compute_ltc(loan_amount: int, total_cost: int) -> float:
        """Loan-to-Cost = Loan / Total Cost."""
        if total_cost <= 0:
            return 0.0
        return round(loan_amount / total_cost, 4)

    @staticmethod
    def compute_equity(value: int, loan_amount: int) -> int:
        """Equity = Value - Loan."""
        return value - loan_amount

    @staticmethod
    def compute_cash_on_cash(noi: int, annual_ds: int, equity: int) -> float:
        """Cash-on-Cash Return = (NOI - DS) / Equity."""
        if equity <= 0:
            return 0.0
        return round((noi - annual_ds) / equity, 4)

    # ── Debt Sizing Constraints ──────────────────────────────

    @staticmethod
    def max_loan_by_ltv(value: int, max_ltv: float) -> int:
        """Maximum loan constrained by LTV."""
        return round(value * max_ltv)

    @staticmethod
    def max_loan_by_dscr(
        noi: int, rate: float, amort_years: int, min_dscr: float
    ) -> int:
        """Maximum loan constrained by DSCR.

        Solve: NOI / (PMT(loan, rate, amort) * 12) >= min_dscr
        => max annual DS = NOI / min_dscr
        => back-solve for loan principal
        """
        if min_dscr <= 0 or rate <= 0 or noi <= 0:
            return 0

        max_annual_ds = noi / min_dscr

        if amort_years == 0:
            # IO: max_ds = loan * rate => loan = max_ds / rate
            return round(max_annual_ds / rate)

        monthly_rate = rate / 12.0
        n_payments = amort_years * 12
        max_monthly = max_annual_ds / 12.0

        if monthly_rate == 0:
            return round(max_monthly * n_payments)

        # Reverse PMT: PV = PMT * [(1+r)^n - 1] / [r * (1+r)^n]
        factor = ((1 + monthly_rate) ** n_payments - 1) / (
            monthly_rate * (1 + monthly_rate) ** n_payments
        )
        return round(max_monthly * factor)

    @staticmethod
    def max_loan_by_debt_yield(noi: int, min_debt_yield: float) -> int:
        """Maximum loan constrained by debt yield.
        DY = NOI / Loan => Loan = NOI / min_DY
        """
        if min_debt_yield <= 0:
            return 0
        return round(noi / min_debt_yield)

    @staticmethod
    def binding_constraint(
        max_ltv_loan: int, max_dscr_loan: int, max_dy_loan: int
    ) -> tuple[int, str]:
        """Return the minimum (binding) loan amount and which constraint binds."""
        constraints = {
            "LTV": max_ltv_loan,
            "DSCR": max_dscr_loan,
            "Debt Yield": max_dy_loan,
        }
        # Filter out zeros
        valid = {k: v for k, v in constraints.items() if v > 0}
        if not valid:
            return 0, "none"
        binding_name = min(valid, key=valid.get)
        return valid[binding_name], binding_name

    # ── Sensitivity Tables ───────────────────────────────────

    @staticmethod
    def sensitivity_rent(
        rent_roll: list[dict], vacancy_rate: float,
        expense_lines: dict[str, int], cap_rate: float,
        annual_ds: int, equity: int, sf: int,
        delta_pct: float = 0.10,
    ) -> dict:
        """Sensitivity to rent changes (+/- delta_pct)."""
        results = {}
        for label, mult in [("up", 1 + delta_pct), ("down", 1 - delta_pct)]:
            adj_pgi = round(sum(t["annual_rent"] for t in rent_roll) * mult)
            adj_egi = round(adj_pgi * (1.0 - vacancy_rate))
            mgmt_fee = expense_lines.get("management_fee", 0)
            if mgmt_fee > 0 and adj_egi > 0:
                # Recalc management fee on new EGI
                orig_egi = round(sum(t["annual_rent"] for t in rent_roll) * (1.0 - vacancy_rate))
                if orig_egi > 0:
                    mgmt_pct = mgmt_fee / orig_egi
                    adj_mgmt = round(adj_egi * mgmt_pct)
                    adj_expenses = dict(expense_lines)
                    adj_expenses["management_fee"] = adj_mgmt
                    adj_opex = sum(adj_expenses.values())
                else:
                    adj_opex = sum(expense_lines.values())
            else:
                adj_opex = sum(expense_lines.values())
            adj_noi = adj_egi - adj_opex
            adj_value = round(adj_noi / cap_rate) if cap_rate > 0 else 0
            adj_dscr = round(adj_noi / annual_ds, 2) if annual_ds > 0 else 0.0
            adj_coc = round((adj_noi - annual_ds) / equity, 4) if equity > 0 else 0.0
            results[f"rent_{label}_{int(delta_pct*100)}pct"] = {
                "noi": adj_noi,
                "value": adj_value,
                "dscr": adj_dscr,
                "cash_on_cash": adj_coc,
            }
        return results

    @staticmethod
    def sensitivity_vacancy(
        pgi: int, expense_lines: dict[str, int],
        cap_rate: float, annual_ds: int, equity: int,
        base_vacancy: float, delta_pct: float = 0.05,
    ) -> dict:
        """Sensitivity to vacancy changes (+/- delta_pct absolute)."""
        results = {}
        for label, adj in [("up", base_vacancy + delta_pct), ("down", max(0, base_vacancy - delta_pct))]:
            adj_egi = round(pgi * (1.0 - adj))
            adj_opex = sum(expense_lines.values())
            adj_noi = adj_egi - adj_opex
            adj_value = round(adj_noi / cap_rate) if cap_rate > 0 else 0
            adj_dscr = round(adj_noi / annual_ds, 2) if annual_ds > 0 else 0.0
            adj_coc = round((adj_noi - annual_ds) / equity, 4) if equity > 0 else 0.0
            results[f"vacancy_{label}_{int(delta_pct*100)}pct"] = {
                "vacancy_rate": round(adj, 3),
                "noi": adj_noi,
                "value": adj_value,
                "dscr": adj_dscr,
                "cash_on_cash": adj_coc,
            }
        return results

    @staticmethod
    def sensitivity_cap_rate(
        noi: int, annual_ds: int, equity: int, sf: int,
        base_cap: float, delta_bps: int = 50,
    ) -> dict:
        """Sensitivity to cap rate changes (+/- delta_bps)."""
        results = {}
        delta = delta_bps / 10000.0
        for label, adj_cap in [("up", base_cap + delta), ("down", max(0.01, base_cap - delta))]:
            adj_value = round(noi / adj_cap) if adj_cap > 0 else 0
            adj_coc = round((noi - annual_ds) / equity, 4) if equity > 0 else 0.0
            results[f"cap_{label}_{delta_bps}bps"] = {
                "cap_rate": round(adj_cap, 4),
                "value": adj_value,
                "price_per_sf": round(adj_value / sf, 2) if sf > 0 else 0.0,
                "cash_on_cash": adj_coc,
            }
        return results

    # ── 1031 Exchange & Taxation ─────────────────────────────

    @staticmethod
    def compute_annual_depreciation(basis: int, useful_life_years: float) -> int:
        """Straight-line annual depreciation = basis / useful life."""
        if useful_life_years <= 0:
            return 0
        return round(basis / useful_life_years)

    @staticmethod
    def compute_accumulated_depreciation(
        basis: int, useful_life_years: float, years_held: int
    ) -> int:
        """Accumulated depreciation = annual depr * years held (capped at basis)."""
        if useful_life_years <= 0:
            return 0
        annual = basis / useful_life_years
        return round(min(annual * years_held, basis))

    @staticmethod
    def compute_adjusted_basis(original_basis: int, accumulated_depreciation: int) -> int:
        """Adjusted basis = original basis - accumulated depreciation."""
        return max(0, original_basis - accumulated_depreciation)

    @staticmethod
    def compute_total_gain(sale_price: int, adjusted_basis: int) -> int:
        """Total gain on sale = sale price - adjusted basis."""
        return sale_price - adjusted_basis

    @staticmethod
    def compute_depreciation_recapture(accumulated_depreciation: int, total_gain: int) -> int:
        """Section 1250 depreciation recapture = min(accumulated depr, total gain).
        Taxed at 25% federal rate.
        """
        if total_gain <= 0:
            return 0
        return min(accumulated_depreciation, total_gain)

    @staticmethod
    def compute_capital_gain_above_recapture(total_gain: int, recapture_amount: int) -> int:
        """Long-term capital gain above recapture = total gain - recapture.
        Taxed at 20% federal rate + 3.8% NIIT.
        """
        return max(0, total_gain - recapture_amount)

    @staticmethod
    def compute_tax_on_sale(
        total_gain: int,
        recapture_amount: int,
        recapture_rate: float = 0.25,
        ltcg_rate: float = 0.20,
        niit_rate: float = 0.038,
    ) -> dict:
        """Compute full tax liability on a CRE sale (no 1031 exchange).

        Returns dict with recapture_tax, ltcg_tax, niit_tax, total_tax.
        """
        if total_gain <= 0:
            return {"recapture_tax": 0, "ltcg_tax": 0, "niit_tax": 0, "total_tax": 0}

        ltcg_amount = max(0, total_gain - recapture_amount)
        recapture_tax = round(recapture_amount * recapture_rate)
        ltcg_tax = round(ltcg_amount * ltcg_rate)
        niit_tax = round(total_gain * niit_rate)
        total_tax = recapture_tax + ltcg_tax + niit_tax

        return {
            "recapture_tax": recapture_tax,
            "ltcg_tax": ltcg_tax,
            "niit_tax": niit_tax,
            "total_tax": total_tax,
        }

    @staticmethod
    def compute_1031_boot(
        relinquished_debt: int,
        replacement_debt: int,
        cash_received: int = 0,
    ) -> dict:
        """Compute boot in a 1031 exchange.

        Mortgage boot = debt relief - replacement debt (if positive).
        Cash boot = cash received from exchange.
        Total boot = mortgage_boot + cash_boot (taxable).
        """
        mortgage_boot = max(0, relinquished_debt - replacement_debt)
        total_boot = mortgage_boot + cash_received
        return {
            "mortgage_boot": mortgage_boot,
            "cash_boot": cash_received,
            "total_boot": total_boot,
        }

    @staticmethod
    def compute_1031_gain_recognition(
        total_gain: int,
        total_boot: int,
    ) -> dict:
        """In a 1031 exchange, recognized gain = lesser of total gain or boot received.
        Deferred gain = total gain - recognized gain.
        """
        recognized_gain = min(max(0, total_gain), max(0, total_boot))
        deferred_gain = max(0, total_gain - recognized_gain)
        return {
            "recognized_gain": recognized_gain,
            "deferred_gain": deferred_gain,
        }

    @staticmethod
    def compute_1031_new_basis(
        replacement_cost: int,
        deferred_gain: int,
    ) -> int:
        """New basis in replacement property = cost - deferred gain.
        (Carries over the deferred gain as reduced basis.)
        """
        return max(0, replacement_cost - deferred_gain)

    @staticmethod
    def compute_cost_segregation(
        building_basis: int,
        accel_pct: float,
        bonus_depreciation_pct: float = 0.60,
    ) -> dict:
        """Cost segregation study results.

        accel_pct: fraction of basis reclassified to shorter-life (5/7/15 yr).
        bonus_depreciation_pct: current bonus depreciation rate.

        Returns dict with accelerated_basis, remaining_39yr_basis,
        year_1_bonus_deduction, annual_39yr_deduction.
        """
        accelerated_basis = round(building_basis * accel_pct)
        remaining_39yr = building_basis - accelerated_basis
        year_1_bonus = round(accelerated_basis * bonus_depreciation_pct)
        annual_39yr = round(remaining_39yr / 39) if remaining_39yr > 0 else 0

        return {
            "accelerated_basis": accelerated_basis,
            "remaining_39yr_basis": remaining_39yr,
            "year_1_bonus_deduction": year_1_bonus,
            "annual_39yr_deduction": annual_39yr,
        }

    # ── Full Gold Computation ────────────────────────────────

    @classmethod
    def compute_gold(cls, deal: dict) -> dict:
        """Compute all gold-standard underwriting values for a deal skeleton.

        Args:
            deal: dict with keys: rent_roll, vacancy_rate, expense_lines,
                  cap_rate, sf, debt (loan_amount, rate, amort_years, io_years)

        Returns:
            dict with all computed metrics for gold.numeric_targets
        """
        rent_roll = deal["rent_roll"]
        vacancy_rate = deal["vacancy_rate"]
        expense_lines = deal["expense_lines"]
        cap_rate = deal["cap_rate"]
        sf = deal["sf"]
        debt = deal["debt"]

        # Revenue chain
        pgi = cls.compute_pgi(rent_roll)
        vacancy_loss = cls.compute_vacancy_loss(pgi, vacancy_rate)
        egi = cls.compute_egi(pgi, vacancy_rate)

        # Recalc management fee on actual EGI
        mgmt_pct = deal.get("management_fee_pct", 0.04)
        mgmt_fee = cls.compute_management_fee(egi, mgmt_pct)
        final_expenses = dict(expense_lines)
        final_expenses["management_fee"] = mgmt_fee
        total_opex = cls.compute_total_opex(final_expenses)

        # NOI
        noi = cls.compute_noi(egi, total_opex)
        noi_per_sf = cls.compute_noi_per_sf(noi, sf)

        # Valuation
        value = cls.compute_value(noi, cap_rate)
        price_per_sf = cls.compute_price_per_sf(value, sf)

        # Debt metrics
        loan_amount = debt["loan_amount"]
        rate = debt["rate"]
        amort_years = debt["amort_years"]
        io_years = debt.get("io_years", 0)

        annual_ds = cls.compute_annual_debt_service(loan_amount, rate, amort_years, io_years)
        dscr = cls.compute_dscr(noi, annual_ds)
        debt_yield = cls.compute_debt_yield(noi, loan_amount)
        ltv = cls.compute_ltv(loan_amount, value)
        equity = cls.compute_equity(value, loan_amount)
        cash_on_cash = cls.compute_cash_on_cash(noi, annual_ds, equity)

        # Sensitivity
        sensitivity = {}
        sensitivity.update(cls.sensitivity_rent(
            rent_roll, vacancy_rate, final_expenses, cap_rate, annual_ds, equity, sf
        ))
        sensitivity.update(cls.sensitivity_vacancy(
            pgi, final_expenses, cap_rate, annual_ds, equity, vacancy_rate
        ))
        sensitivity.update(cls.sensitivity_cap_rate(
            noi, annual_ds, equity, sf, cap_rate
        ))

        return {
            "pgi": pgi,
            "vacancy_loss": vacancy_loss,
            "egi": egi,
            "total_opex": total_opex,
            "management_fee": mgmt_fee,
            "noi": noi,
            "noi_per_sf": noi_per_sf,
            "value": value,
            "price_per_sf": price_per_sf,
            "cap_rate": cap_rate,
            "loan_amount": loan_amount,
            "annual_debt_service": annual_ds,
            "dscr": dscr,
            "debt_yield": debt_yield,
            "ltv": ltv,
            "equity": equity,
            "cash_on_cash": cash_on_cash,
            "expense_lines": final_expenses,
            "sensitivity": sensitivity,
        }
