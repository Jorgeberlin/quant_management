# Quant Management

Repository for the **Quantitative Management** project, covering risk analysis, factor analysis and attribution, portfolio optimization, and backtesting.

**Authors:** Xabier, Jorge Peñaranda 

## Project Structure

```text
quant-management/
├── README.md
├── pyproject.toml
├── quantmgmt/
│   ├── data/          # Data download and management
│   ├── risk/          # Risk metrics and analysis
│   ├── utils/         # Risk summary table
│   ├── analytics/     # Factor analysis and attribution
│   ├── optimizers/    # Portfolio optimization
│   └── backtest/      # Backtesting
├── notebooks/         # Practical applications
├── tests/             # Unit tests
└── data/
    └── cache/         # Local data cache
```

## Installation

```bash
git clone https://github.com/Jorgeberlin/quant_management.git
cd quant_management

python -m venv .venv
```

Activate the virtual environment:

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

Install the project (with test dependencies):

```bash
pip install -e ".[dev]"
```

## Usage

All risk metrics take **prices** (`pd.Series` or `pd.DataFrame`, indexed by date) and a
`periods_per_year` argument (252 daily, 52 weekly, 12 monthly; default 252).
With a DataFrame they return one value per column, so different assets, portfolios
or NAVs can be compared directly.

```python
from quantmgmt.data import YahooFinanceData
from quantmgmt.risk import sharpe_ratio, max_drawdown, tail_adjusted_volatility
from quantmgmt.utils import generate_risk_summary, format_risk_summary

prices = YahooFinanceData().get_prices(["SPY", "TLT", "GLD"], "2014-01-01", "2025-12-31")

sharpe_ratio(prices, periods_per_year=252, risk_free_rate=0.02)
max_drawdown(prices)
tail_adjusted_volatility(prices)   # custom metric, see quantmgmt/risk/custom.py

summary = generate_risk_summary(prices, risk_free_rate=0.02)
format_risk_summary(summary, transpose=True)
```

Prices are adjusted for splits and dividends and cached in `data/cache/`.
See `notebooks/01_riesgo.ipynb` for the full application.

## Practices

* **P1 — Risk Management**
* **P2 — Factor Analysis & Attribution**
* **P3 — Portfolio Optimization**
* **P4+ — Backtesting and further methodologies**

## Testing

Run the test suite with:

```bash
pytest
```
