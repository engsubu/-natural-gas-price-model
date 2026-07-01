# JPMorgan Task 1: Natural Gas Price Analysis
    **Approach**: Monthly EOD gas prices 2020-10 to 2024-09. Used Holt-Winters with 12-month seasonality to extrapolate +12 months.
    **Run**: `pip install -r requirements.txt` then `python src/price_estimator.py`
    **Insights**: Prices show winter peaks due to heating demand.
