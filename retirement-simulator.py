import streamlit as st
st.title("Monte Carlo Retirement Simulator (Mid-Year Convention)")

st.markdown('''
### How This Model Works (Updated — Single-Country Framework)
- **Purpose:** This simulator estimates retirement outcomes under uncertainty using Monte Carlo simulation. It models investment returns, income, spending, Social Security, taxes, and portfolio withdrawals across your lifetime.
- **Two-Portfolios:** You are able to model two separate portfolios — a **liquid/taxable** portfolio (available immediately) and a **retirement** portfolio (available only after a chosen access age).
- **Spending:** You provide a single annual spending amount in today's dollars. This amount inflates each year using your chosen spending inflation rate.
- **Income & Savings:** You may enter a pre-retirement income amount. Any surplus income (after taxes and spending) is automatically added to the liquid portfolio.
- **Social Security (SSI):** SSI is entered in **today's dollars** and grows continuously each year using the chosen COLA rate. SSI is only added to your income once you reach your start age.
- **Taxes:** You can define effective tax rates for both working years and retirement years. Taxes reduce income before spending is applied. If withdrawals are required during retirement, a partial tax adjustment is applied to approximate real-world taxation of withdrawals.
- **Lump Sum Events:** You may include a lump sum to be received in a future year (inheritance, asset sale, etc.). This value is **not inflated** and is added directly to your liquid portfolio in the year received.
- **Mid-Year Convention:** Each year’s investment return is applied half before and half after cash flows (income/spending), creating more realistic compounding behavior.
- **Investments:** You define return assumptions and volatility for equities, bonds, and cash. Correlations among asset classes are modeled using a Cholesky decomposition.
- **Glide Paths:** You define separate asset allocations for each portfolio, both **before** and **after** the retirement-access age.
''')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.cm as cm

MAX_SIMS = 20000

# --- Inputs ---
liquid_init = float(st.text_input("Liquid / taxable portfolio ($)", value="600000"))
retirement_init = float(st.text_input("Retirement portfolio ($)", value="400000"))

# Unified Age & Economic Inputs
current_age = int(st.text_input("Current age", value="40"))
retire_age = int(st.text_input("Retirement age", value="65"))
death_age = int(st.text_input("Age at death", value="100"))

# Unified Spending & Income Inputs
annual_spending = float(st.text_input("Annual spending ($)", value="30000"))
spending_inflation_rate = float(st.text_input("Annual spending inflation rate (%)", value="4.0")) / 100
annual_income = float(st.text_input("Annual earned income before retirement ($)", value="0"))

# Investment Return Assumptions
mean_equity = float(st.text_input("Mean annual return for equities (%)", value="8.5")) / 100
std_equity = float(st.text_input("Volatility for equities (%)", value="17.0")) / 100
mean_bonds = float(st.text_input("Mean annual return for bonds (%)", value="5.0")) / 100
std_bonds = float(st.text_input("Volatility for bonds (%)", value="6.0")) / 100
mean_cash = float(st.text_input("Mean annual return for cash (%)", value="2.5")) / 100
std_cash = float(st.text_input("Volatility for cash (%)", value="1.5")) / 100

# Portfolio Allocation Inputs
st.subheader("Liquid Portfolio Allocation — Before and After Access Age")
liq_eq_pre = float(st.text_input("Liquid portfolio equity allocation (%) before access age", value="65")) / 100
liq_bd_pre = float(st.text_input("Liquid portfolio bond allocation (%) before access age", value="25")) / 100
liq_cs_pre = float(st.text_input("Liquid portfolio cash allocation (%) before access age", value="10")) / 100

liq_eq_post = float(st.text_input("Liquid portfolio equity allocation (%) at/after access age", value="40")) / 100
liq_bd_post = float(st.text_input("Liquid portfolio bond allocation (%) at/after access age", value="50")) / 100
liq_cs_post = float(st.text_input("Liquid portfolio cash allocation (%) at/after access age", value="10")) / 100

# Validate liquid allocations
if abs(liq_eq_pre + liq_bd_pre + liq_cs_pre - 1) > 1e-6:
    st.error("Liquid portfolio allocations before access age must sum to 100%.")
    st.stop()

if abs(liq_eq_post + liq_bd_post + liq_cs_post - 1) > 1e-6:
    st.error("Liquid portfolio allocations at/after access age must sum to 100%.")
    st.stop()

st.subheader("Retirement Portfolio Allocation — Before and After Access Age")
ret_eq_pre = float(st.text_input("Retirement portfolio equity allocation (%) before access age", value="70")) / 100
ret_bd_pre = float(st.text_input("Retirement portfolio bond allocation (%) before access age", value="25")) / 100
ret_cs_pre = float(st.text_input("Retirement portfolio cash allocation (%) before access age", value="5")) / 100

ret_eq_post = float(st.text_input("Retirement portfolio equity allocation (%) at/after access age", value="50")) / 100
ret_bd_post = float(st.text_input("Retirement portfolio bond allocation (%) at/after access age", value="40")) / 100
ret_cs_post = float(st.text_input("Retirement portfolio cash allocation (%) at/after access age", value="10")) / 100

# Validate retirement allocations
if abs(ret_eq_pre + ret_bd_pre + ret_cs_pre - 1) > 1e-6:
    st.error("Retirement portfolio allocations before access age must sum to 100%.")
    st.stop()

if abs(ret_eq_post + ret_bd_post + ret_cs_post - 1) > 1e-6:
    st.error("Retirement portfolio allocations at/after access age must sum to 100%.")
    st.stop()

# Tax rates
work_tax_rate = float(st.text_input("Effective tax rate while working (%)", value="20.0")) / 100
retire_tax_rate = float(st.text_input("Effective tax rate during retirement (%)", value="15.0")) / 100

# SSI / Pension
start_ssi_age = int(st.text_input("Age to start receiving retirement income (e.g., SSI)", value="67"))
ssi_amount_today = float(st.text_input("Annual retirement income in today's dollars ($)", value="26000"))
ssi_inflation_rate = float(st.text_input("SSI annual inflation rate (COLA %)", value="2.0")) / 100
include_ssi_taxable = st.checkbox("Include SSI in taxable income?", value=False)

# Retirement access age & early withdrawal penalty
retirement_access_age = int(st.text_input("Age when retirement portfolio becomes accessible", value="60"))
early_withdraw_penalty_rate = float(st.text_input("Early withdrawal penalty rate on retirement account before access age (%)", value="10.0")) / 100

# Lump Sum Event
receive_lump_age = int(st.text_input("Age when lump sum is received", value="70"))
lump_amount_today = float(st.text_input("Lump sum amount ($)", value="100000"))

# --- Monte Carlo Engine ---

@st.cache_data(show_spinner=False)
def run_simulation(
    annual_spending_override=None,
    override_liq_alloc=None,
    override_ret_alloc=None
):
    """
    Core Monte Carlo simulation engine.
    Optional overrides allow reuse for sensitivity and allocation analyses.
    """

    spend_base = annual_spending_override if annual_spending_override is not None else annual_spending

    results, final_balances = [], []

    for _ in range(sims):
        liquid_balance = liquid_init
        retirement_balance = retirement_init
        path = []

        for year in range(total_years):
            age = current_age + year

            spend = spend_base * ((1 + spending_inflation_rate) ** (age - current_age))
            income = annual_income if age < retire_age else 0
            tax_rate = retire_tax_rate if age >= retire_age else work_tax_rate

            taxable_income = 0
            if age >= start_ssi_age:
                years_since_now = age - current_age
                inflated_ssi = ssi_amount_today * ((1 + ssi_inflation_rate) ** years_since_now)
                income += inflated_ssi
                if include_ssi_taxable:
                    taxable_income += inflated_ssi

            if age == receive_lump_age:
                liquid_balance += lump_amount_today

            taxable_income += income
            tax = taxable_income * tax_rate
            income_after_tax = income - tax

            if age >= retire_age and income_after_tax < spend:
                withdraw_needed_tax_basis = spend - income_after_tax
                tax_adj = 1 + (retire_tax_rate * 0.5)
                spend = income_after_tax + withdraw_needed_tax_basis * tax_adj

            rand = np.random.normal(0, 1, 3)
            correlated = chol @ rand
            eq_ret = np.exp(mu_eq_ln + sigma_eq_ln * correlated[0]) - 1
            bd_ret = mean_bonds + correlated[1] * std_bonds
            cs_ret = np.maximum(0, mean_cash + correlated[2] * std_cash)

            if override_liq_alloc is None:
                if age < retirement_access_age:
                    w_liq_eq, w_liq_bd, w_liq_cs = liq_eq_pre, liq_bd_pre, liq_cs_pre
                else:
                    w_liq_eq, w_liq_bd, w_liq_cs = liq_eq_post, liq_bd_post, liq_cs_post
            else:
                w_liq_eq, w_liq_bd, w_liq_cs = override_liq_alloc

            if override_ret_alloc is None:
                if age < retirement_access_age:
                    w_ret_eq, w_ret_bd, w_ret_cs = ret_eq_pre, ret_bd_pre, ret_cs_pre
                else:
                    w_ret_eq, w_ret_bd, w_ret_cs = ret_eq_post, ret_bd_post, ret_cs_post
            else:
                w_ret_eq, w_ret_bd, w_ret_cs = override_ret_alloc

            liq_growth = w_liq_eq * eq_ret + w_liq_bd * bd_ret + w_liq_cs * cs_ret
            ret_growth = w_ret_eq * eq_ret + w_ret_bd * bd_ret + w_ret_cs * cs_ret

            mid_liq = (1 + liq_growth) ** 0.5 - 1
            mid_ret = (1 + ret_growth) ** 0.5 - 1

            liquid_balance *= (1 + mid_liq)
            retirement_balance *= (1 + mid_ret)

            surplus = max(0, income_after_tax - spend)
            deficit = max(0, spend - income_after_tax)

            if surplus > 0:
                liquid_balance += surplus

            if deficit > 0:
                withdraw_remaining = deficit
                if liquid_balance >= withdraw_remaining:
                    liquid_balance -= withdraw_remaining
                else:
                    withdraw_remaining -= liquid_balance
                    liquid_balance = 0
                if withdraw_remaining > 0:
                    if age < retirement_access_age:
                        gross_from_ret = withdraw_remaining * (1 + early_withdraw_penalty_rate)
                    else:
                        gross_from_ret = withdraw_remaining
                    retirement_balance -= gross_from_ret

            liquid_balance *= (1 + mid_liq)
            retirement_balance *= (1 + mid_ret)

            path.append(liquid_balance + retirement_balance)

        results.append(path)
        final_balances.append(path[-1])

    return np.array(results), np.array(final_balances)


# --- Run Base Simulation ---
results, final_balances = run_simulation()

# --- Results ---
median_final = np.median(final_balances)
p10, p20, p80, p90 = np.percentile(final_balances, [10, 20, 80, 90])
min_final, max_final = np.min(final_balances), np.max(final_balances)
success_rate = np.mean(np.array(final_balances) > 0) * 100

# --- Visualizations ---

# --- Visualizations ---

# Create tabs for visual outputs
tabs = st.tabs(["Overview", "Distributions", "Stress Scenarios", "Sensitivity", "Allocation Heatmap", "Asset Contributions"])(["Overview", "Distributions", "Stress Scenarios", "Sensitivity", "Allocation Heatmap"])(["Overview", "Distributions", "Stress Scenarios", "Sensitivity"])(["Overview", "Distributions", "Stress Scenarios"])

tab_overview = tabs[0]
tab_dist = tabs[1]
tab_stress = tabs[2]
tab_sens = tabs[3]
tab_alloc = tabs[4]
tab_contrib = tabs[5]

# --- Overview Tab (Fan Chart) ---
with tab_overview:
    st.markdown("ℹ️ **Fan Chart:** Shaded bands show the range of possible portfolio paths over time. The darker band represents the middle 60% of outcomes (20–80%), while the lighter band shows more extreme scenarios (10–90%).")
    st.subheader("Portfolio Projection — Confidence Bands")

    years = np.arange(total_years)

    # Compute percentiles across simulations
    p10_path = np.percentile(results, 10, axis=0)
    p20_path = np.percentile(results, 20, axis=0)
    p50_path = np.percentile(results, 50, axis=0)
    p80_path = np.percentile(results, 80, axis=0)
    p90_path = np.percentile(results, 90, axis=0)

    plt.figure(figsize=(10, 6))

    # Confidence bands
    plt.fill_between(years, p10_path, p90_path, color='steelblue', alpha=0.15, label='10–90 Percentile')
    plt.fill_between(years, p20_path, p80_path, color='steelblue', alpha=0.30, label='20–80 Percentile')

    # Median path
    plt.plot(years, p50_path, color='black', linewidth=2.5, label='Median Path')

    plt.title("Monte Carlo Retirement Projection — Confidence Bands")
    plt.xlabel("Years from Current Age")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)
    plt.legend()
    plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_formatter))

    st.pyplot(plt)

# --- Distributions Tab ---
with tab_dist:
    st.markdown("ℹ️ **Ending Balance Distribution:** This histogram shows the probability distribution of portfolio values at death. It highlights downside risk, upside potential, and skewness.")
    st.subheader("Ending Portfolio Balance Distribution")

    plt.figure(figsize=(10, 6))
    plt.hist(final_balances, bins=50, density=True, color='slateblue', alpha=0.75)

    plt.axvline(median_final, color='black', linewidth=2, linestyle='--', label='Median')
    plt.axvline(p20, color='gray', linestyle=':', label='20th / 80th Percentile')
    plt.axvline(p80, color='gray', linestyle=':')

    plt.title("Distribution of Portfolio Value at Death")
    plt.xlabel("Portfolio Value ($)")
    plt.ylabel("Density")
    plt.legend()
    plt.gca().xaxis.set_major_formatter(FuncFormatter(currency_formatter))

    st.pyplot(plt)

# --- Stress Scenarios Tab ---
with tab_stress:
    st.markdown("ℹ️ **Sequence-of-Returns Risk:** These paths isolate the impact of market returns in the *early* years. Poor early returns can permanently impair sustainability even if long-term averages are similar.")
    st.subheader("Sequence-of-Returns Stress Scenarios")
    st.markdown("This view highlights how *early* market outcomes shape long-term results. We compare paths with the **worst** and **best** early-period performance against the median.")

    # Define early window (first 5 years)
    early_years = min(5, total_years)

    # Rank simulations by cumulative return over early window
    early_cum = results[:, early_years - 1]
    order = np.argsort(early_cum)

    worst_idx = order[: max(1, sims // 20)]   # worst 5%
    best_idx = order[-max(1, sims // 20):]    # best 5%

    median_path = np.median(results, axis=0)

    plt.figure(figsize=(10, 6))

    # Plot worst early-return paths
    for i in worst_idx:
        plt.plot(results[i], color='crimson', alpha=0.25)

    # Plot best early-return paths
    for i in best_idx:
        plt.plot(results[i], color='seagreen', alpha=0.25)

    # Median path
    plt.plot(median_path, color='black', linewidth=2.5, label='Median Path')

    plt.title("Sequence-of-Returns Stress Test (Early-Period Outcomes)")
    plt.xlabel("Years from Current Age")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)
    plt.legend(["Worst Early Returns (Bottom 5%)", "Best Early Returns (Top 5%)", "Median"], loc='best')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_formatter))

    st.pyplot(plt)

# --- Sensitivity Tab ---
with tab_sens:
    st.markdown("ℹ️ **Spending Sensitivity:** Each point reruns the full Monte Carlo simulation at a different spending level. Nonlinear drops reveal spending cliffs where sustainability rapidly deteriorates.")
    st.subheader("Spending Sensitivity Curve")
    st.markdown("This analysis shows how sensitive retirement outcomes are to changes in annual spending assumptions. Each point represents a full Monte Carlo simulation run.")

    run_sens = st.button("Run Spending Sensitivity Analysis")

    if run_sens:
        spend_range = np.linspace(annual_spending * 0.7, annual_spending * 1.3, 9)
        p20_vals, p50_vals, p80_vals = [], [], []

        for spend_test in spend_range:
            _, finals = run_simulation(annual_spending_override=spend_test)
            p20_vals.append(np.percentile(finals, 20))
            p50_vals.append(np.percentile(finals, 50))
            p80_vals.append(np.percentile(finals, 80))

        plt.figure(figsize=(10, 6))
        plt.plot(spend_range, p50_vals, label="Median", color="black", linewidth=2)
        plt.fill_between(spend_range, p20_vals, p80_vals, alpha=0.3, label="20–80% Band")
        plt.xlabel("Annual Spending (Today's $)")
        plt.ylabel("Portfolio at Death ($)")
        plt.title("Spending Sensitivity Curve")
        plt.legend()
        plt.grid(True)
        plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_formatter))
        st.pyplot(plt)

# --- Allocation Heatmap Tab ---
with tab_alloc:
    st.markdown("ℹ️ **Allocation Survival Heatmap:** Colors represent the probability of finishing with a positive balance. Broad green regions indicate robust allocations; narrow ridges indicate fragile strategies.")
    st.subheader("Allocation Survival Heatmap")
    st.markdown("This analysis shows how **robust** different equity/bond allocations are by measuring the probability of finishing with a positive portfolio balance. Each cell represents a full Monte Carlo run set.")

    run_alloc = st.button("Run Allocation Heatmap Analysis")

    if run_alloc:
        eq_weights = np.linspace(0.2, 0.8, 7)
        survival = np.zeros((len(eq_weights), len(eq_weights)))

        for i, w_liq_eq in enumerate(eq_weights):
            for j, w_ret_eq in enumerate(eq_weights):
                w_liq = (w_liq_eq, 1 - w_liq_eq, 0)
                w_ret = (w_ret_eq, 1 - w_ret_eq, 0)
                _, finals = run_simulation(
                    override_liq_alloc=w_liq,
                    override_ret_alloc=w_ret
                )
                survival[i, j] = np.mean(finals > 0)

        plt.figure(figsize=(8, 6))
        plt.imshow(survival, origin='lower', cmap='viridis', aspect='auto')
        plt.colorbar(label='Survival Probability')
        plt.xticks(range(len(eq_weights)), [f"{int(w*100)}%" for w in eq_weights])
        plt.yticks(range(len(eq_weights)), [f"{int(w*100)}%" for w in eq_weights])
        plt.xlabel("Retirement Portfolio Equity Allocation")
        plt.ylabel("Liquid Portfolio Equity Allocation")
        plt.title("Allocation Survival Heatmap")
        st.pyplot(plt)


with tab_alloc:
    st.subheader("Allocation Survival Heatmap")
    st.markdown("This analysis shows how **robust** different equity/bond allocations are by measuring the probability of finishing with a positive portfolio balance. Each cell represents a full Monte Carlo run set.")

    run_alloc = st.button("Run Allocation Heatmap Analysis")

    if run_alloc:
        eq_weights = np.linspace(0.2, 0.8, 7)
        bd_weights = 1 - eq_weights

        survival = np.zeros((len(eq_weights), len(eq_weights)))

        for i, w_eq in enumerate(eq_weights):
            for j, w_eq_ret in enumerate(eq_weights):
                finals = []
                for _ in range(sims):
                    liquid_balance = liquid_init
                    retirement_balance = retirement_init

                    for year in range(total_years):
                        age = current_age + year
                        spend = annual_spending * ((1 + spending_inflation_rate) ** (age - current_age))
                        income = annual_income if age < retire_age else 0
                        tax_rate = retire_tax_rate if age >= retire_age else work_tax_rate

                        taxable_income = 0
                        if age >= start_ssi_age:
                            years_since_now = age - current_age
                            inflated_ssi = ssi_amount_today * ((1 + ssi_inflation_rate) ** years_since_now)
                            income += inflated_ssi
                            if include_ssi_taxable:
                                taxable_income += inflated_ssi

                        taxable_income += income
                        tax = taxable_income * tax_rate
                        income_after_tax = income - tax

                        if age >= retire_age and income_after_tax < spend:
                            withdraw_needed_tax_basis = spend - income_after_tax
                            tax_adj = 1 + (retire_tax_rate * 0.5)
                            spend = income_after_tax + withdraw_needed_tax_basis * tax_adj

                        rand = np.random.normal(0, 1, 3)
                        correlated = chol @ rand
                        eq_ret = np.exp(mu_eq_ln + sigma_eq_ln * correlated[0]) - 1
                        bd_ret = mean_bonds + correlated[1] * std_bonds
                        cs_ret = np.maximum(0, mean_cash + correlated[2] * std_cash)

                        liq_growth = w_eq * eq_ret + bd_weights[i] * bd_ret
                        ret_growth = w_eq_ret * eq_ret + bd_weights[j] * bd_ret

                        mid_liq = (1 + liq_growth) ** 0.5 - 1
                        mid_ret = (1 + ret_growth) ** 0.5 - 1

                        liquid_balance *= (1 + mid_liq)
                        retirement_balance *= (1 + mid_ret)

                        surplus = max(0, income_after_tax - spend)
                        deficit = max(0, spend - income_after_tax)

                        if surplus > 0:
                            liquid_balance += surplus

                        if deficit > 0:
                            withdraw_remaining = deficit
                            if liquid_balance >= withdraw_remaining:
                                liquid_balance -= withdraw_remaining
                                withdraw_remaining = 0
                            else:
                                withdraw_remaining -= liquid_balance
                                liquid_balance = 0
                            if withdraw_remaining > 0:
                                if age < retirement_access_age:
                                    gross_from_ret = withdraw_remaining * (1 + early_withdraw_penalty_rate)
                                else:
                                    gross_from_ret = withdraw_remaining
                                retirement_balance -= gross_from_ret

                        liquid_balance *= (1 + mid_liq)
                        retirement_balance *= (1 + mid_ret)

                    finals.append(liquid_balance + retirement_balance)

                survival[i, j] = np.mean(np.array(finals) > 0)

        plt.figure(figsize=(8, 6))
        plt.imshow(survival, origin='lower', cmap='viridis', aspect='auto')
        plt.colorbar(label='Survival Probability')
        plt.xticks(range(len(eq_weights)), [f"{int(w*100)}%" for w in eq_weights])
        plt.yticks(range(len(eq_weights)), [f"{int(w*100)}%" for w in eq_weights])
        plt.xlabel("Retirement Portfolio Equity Allocation")
        plt.ylabel("Liquid Portfolio Equity Allocation")
        plt.title("Allocation Survival Heatmap")
        st.pyplot(plt)

# --- Asset Contribution Breakdown Tab ---
with tab_contrib:
    st.markdown("ℹ️ **Asset Contribution Breakdown:** Bars show median contributions to ending wealth. This is a diagnostic attribution, not a precise accounting identity, due to compounding interactions.")
    st.subheader("Asset Contribution Breakdown")
    st.markdown("This view decomposes **ending portfolio value** into contributions from equities, bonds, cash, net savings (income minus spending), and lump sums. Results shown are **medians across simulations**.")

    # Track contributions (median-based approximation)
    contrib_equity = []
    contrib_bonds = []
    contrib_cash = []
    contrib_savings = []
    contrib_lump = []

    for _ in range(sims):
        liquid_balance = liquid_init
        retirement_balance = retirement_init

        eq_c = bd_c = cs_c = sav_c = lump_c = 0.0

        for year in range(total_years):
            age = current_age + year

            spend = annual_spending * ((1 + spending_inflation_rate) ** (age - current_age))
            income = annual_income if age < retire_age else 0
            tax_rate = retire_tax_rate if age >= retire_age else work_tax_rate

            taxable_income = 0
            if age >= start_ssi_age:
                years_since_now = age - current_age
                inflated_ssi = ssi_amount_today * ((1 + ssi_inflation_rate) ** years_since_now)
                income += inflated_ssi
                if include_ssi_taxable:
                    taxable_income += inflated_ssi

            taxable_income += income
            tax = taxable_income * tax_rate
            income_after_tax = income - tax

            if age == receive_lump_age:
                liquid_balance += lump_amount_today
                lump_c += lump_amount_today

            rand = np.random.normal(0, 1, 3)
            correlated = chol @ rand
            eq_ret = np.exp(mu_eq_ln + sigma_eq_ln * correlated[0]) - 1
            bd_ret = mean_bonds + correlated[1] * std_bonds
            cs_ret = np.maximum(0, mean_cash + correlated[2] * std_cash)

            if age < retirement_access_age:
                w_liq_eq, w_liq_bd, w_liq_cs = liq_eq_pre, liq_bd_pre, liq_cs_pre
                w_ret_eq, w_ret_bd, w_ret_cs = ret_eq_pre, ret_bd_pre, ret_cs_pre
            else:
                w_liq_eq, w_liq_bd, w_liq_cs = liq_eq_post, liq_bd_post, liq_cs_post
                w_ret_eq, w_ret_bd, w_ret_cs = ret_eq_post, ret_bd_post, ret_cs_post

            eq_c += (liquid_balance * w_liq_eq + retirement_balance * w_ret_eq) * eq_ret
            bd_c += (liquid_balance * w_liq_bd + retirement_balance * w_ret_bd) * bd_ret
            cs_c += (liquid_balance * w_liq_cs + retirement_balance * w_ret_cs) * cs_ret

            mid_liq = (1 + (w_liq_eq*eq_ret + w_liq_bd*bd_ret + w_liq_cs*cs_ret)) ** 0.5 - 1
            mid_ret = (1 + (w_ret_eq*eq_ret + w_ret_bd*bd_ret + w_ret_cs*cs_ret)) ** 0.5 - 1

            liquid_balance *= (1 + mid_liq)
            retirement_balance *= (1 + mid_ret)

            surplus = max(0, income_after_tax - spend)
            deficit = max(0, spend - income_after_tax)

            if surplus > 0:
                liquid_balance += surplus
                sav_c += surplus

            if deficit > 0:
                withdraw_remaining = deficit
                if liquid_balance >= withdraw_remaining:
                    liquid_balance -= withdraw_remaining
                else:
                    withdraw_remaining -= liquid_balance
                    liquid_balance = 0
                    retirement_balance -= withdraw_remaining

            liquid_balance *= (1 + mid_liq)
            retirement_balance *= (1 + mid_ret)

        contrib_equity.append(eq_c)
        contrib_bonds.append(bd_c)
        contrib_cash.append(cs_c)
        contrib_savings.append(sav_c)
        contrib_lump.append(lump_c)

    labels = ["Equities", "Bonds", "Cash", "Net Savings", "Lump Sums"]
    values = [np.median(contrib_equity), np.median(contrib_bonds), np.median(contrib_cash), np.median(contrib_savings), np.median(contrib_lump)]

    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color='teal')
    plt.ylabel("Contribution to Ending Portfolio ($)")
    plt.title("Median Asset Contribution Breakdown")
    plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    plt.grid(axis='y')

    st.pyplot(plt)

# --- CSV Export ---
st.subheader("Download Results")

# Prepare CSV data
export_df = {
    "Final Balance": final_balances
}

import pandas as pd
export_df = pd.DataFrame(export_df)

csv_data = export_df.to_csv(index=False)

st.download_button(
    label="Download Final Balances (CSV)",
    data=csv_data,
    file_name="retirement_sim_final_balances.csv",
    mime="text/csv"
)

# --- Summary ---
st.subheader("Simulation Results Summary")
st.write(f"**Median Portfolio at Death:** ${median_final:,.0f}")
st.write(f"**10th Percentile:** ${p10:,.0f}")
st.write(f"**20th Percentile:** ${p20:,.0f}")
st.write(f"**80th Percentile:** ${p80:,.0f}")
st.write(f"**90th Percentile:** ${p90:,.0f}")
st.write(f"**Minimum Portfolio at Death:** ${min_final:,.0f}")
st.write(f"**Maximum Portfolio at Death:** ${max_final:,.0f}")
st.write(f"**Success Rate (Final > 0):** {success_rate:.1f}%")
