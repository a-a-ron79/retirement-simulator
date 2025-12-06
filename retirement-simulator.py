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

# --- Monte Carlo setup ---
mu_eq_ln = np.log(1 + mean_equity) - 0.5 * (std_equity ** 2)
sigma_eq_ln = np.sqrt(np.log(1 + (std_equity ** 2)))

corr = np.array([[1.0, 0.2, 0.0], [0.2, 1.0, 0.1], [0.0, 0.1, 1.0]])
chol = np.linalg.cholesky(corr)

sims = int(st.text_input("Number of simulations (max 20,000)", value="1000"))
if sims > MAX_SIMS:
    sims = MAX_SIMS

total_years = death_age - current_age

# --- Simulation ---
results, final_balances = [], []

for _ in range(sims):
    liquid_balance = liquid_init
    retirement_balance = retirement_init
    path = []

    for year in range(total_years):
        age = current_age + year

        # Unified income & spending
        spend = annual_spending * ((1 + spending_inflation_rate) ** (age - current_age))
        income = annual_income if age < retire_age else 0
        tax_rate = retire_tax_rate if age >= retire_age else work_tax_rate

        # SSI income (single continuous inflation model)
        taxable_income = 0
        if age >= start_ssi_age:
            years_since_now = age - current_age
            inflated_ssi = ssi_amount_today * ((1 + ssi_inflation_rate) ** years_since_now)
            income += inflated_ssi
            if include_ssi_taxable:
                taxable_income += inflated_ssi

        # Lump sum income event (goes to liquid portfolio)
        if age == receive_lump_age:
            lump_sum = lump_amount_today  # no inflation adjustment
            liquid_balance += lump_sum

        # Taxes
        taxable_income += income
        tax = taxable_income * tax_rate
        income_after_tax = income - tax

        # Withdrawal tax inflation adjustment in retirement
        if age >= retire_age and (income_after_tax < spend):
            withdraw_needed_tax_basis = spend - income_after_tax
            tax_adj = 1 + (retire_tax_rate * 0.5)
            spend = income_after_tax + withdraw_needed_tax_basis * tax_adj

        # Draw correlated asset returns
        rand = np.random.normal(0, 1, 3)
        correlated = chol @ rand
        eq_ret = np.exp(mu_eq_ln + sigma_eq_ln * correlated[0]) - 1
        bd_ret = mean_bonds + correlated[1] * std_bonds
        cs_ret = np.maximum(0, mean_cash + correlated[2] * std_cash)

        # Choose allocations by age for each portfolio
        if age < retirement_access_age:
            w_liq_eq, w_liq_bd, w_liq_cs = liq_eq_pre, liq_bd_pre, liq_cs_pre
            w_ret_eq, w_ret_bd, w_ret_cs = ret_eq_pre, ret_bd_pre, ret_cs_pre
        else:
            w_liq_eq, w_liq_bd, w_liq_cs = liq_eq_post, liq_bd_post, liq_cs_post
            w_ret_eq, w_ret_bd, w_ret_cs = ret_eq_post, ret_bd_post, ret_cs_post

        # Portfolio-specific growth rates
        liq_growth = w_liq_eq * eq_ret + w_liq_bd * bd_ret + w_liq_cs * cs_ret
        ret_growth = w_ret_eq * eq_ret + w_ret_bd * bd_ret + w_ret_cs * cs_ret

        mid_liq = (1 + liq_growth) ** 0.5 - 1
        mid_ret = (1 + ret_growth) ** 0.5 - 1

        # First half-year growth
        liquid_balance *= (1 + mid_liq)
        retirement_balance *= (1 + mid_ret)

        # Cash flow handling at mid-year
        surplus = max(0, income_after_tax - spend)
        deficit = max(0, spend - income_after_tax)

        # Surplus goes into liquid portfolio
        if surplus > 0:
            liquid_balance += surplus

        # Deficit funded by withdrawals
        if deficit > 0:
            withdraw_remaining = deficit

            # 1) Withdraw from liquid first
            if liquid_balance >= withdraw_remaining:
                liquid_balance -= withdraw_remaining
                withdraw_remaining = 0
            else:
                withdraw_remaining -= liquid_balance
                liquid_balance = 0

            # 2) If still short, tap retirement portfolio
            if withdraw_remaining > 0:
                if age < retirement_access_age:
                    # Early withdrawal with penalty
                    gross_from_ret = withdraw_remaining * (1 + early_withdraw_penalty_rate)
                else:
                    gross_from_ret = withdraw_remaining
                retirement_balance -= gross_from_ret

        # Second half-year growth
        liquid_balance *= (1 + mid_liq)
        retirement_balance *= (1 + mid_ret)

        total_balance = liquid_balance + retirement_balance
        path.append(total_balance)

    results.append(path)
    final_balances.append(path[-1])

results = np.array(results)

# --- Results ---
median_final = np.median(final_balances)
p10, p20, p80, p90 = np.percentile(final_balances, [10, 20, 80, 90])
min_final, max_final = np.min(final_balances), np.max(final_balances)
success_rate = np.mean(np.array(final_balances) > 0) * 100

# --- Plot ---
def currency_formatter(x, pos):
    if x >= 1_000_000:
        return f"${x/1_000_000:.1f}M"
    elif x >= 1_000:
        return f"${x/1_000:.0f}K"
    return f"${x:,.0f}"

plt.figure(figsize=(10, 6))
colors = cm.get_cmap('Spectral', sims)
for i, path in enumerate(results):
    plt.plot(path, color=colors(i / sims), alpha=0.3)

median_path = np.median(results, axis=0)
plt.plot(median_path, color='black', linewidth=2.5, label='Median Path')
plt.legend()

plt.title("Monte Carlo Retirement Projection")
plt.xlabel("Years from Current Age")
plt.ylabel("Portfolio Value ($K / $M)")
plt.grid(True)
plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_formatter))
st.pyplot(plt)

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
