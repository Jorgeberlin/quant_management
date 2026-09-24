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
git clone <https://github.com/Jorgeberlin/quant_management.git>
cd quant-management

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

Install the project:

```bash
pip install -e .
```

## Usage

Example:

```python
from quantmgmt.risk import sharpe_ratio

sharpe = sharpe_ratio(returns, periods_per_year=252)
print(sharpe)
```

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
