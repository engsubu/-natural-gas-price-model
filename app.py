import pandas as pd
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from datetime import datetime

    def load_and_fit(path='../data/gas_prices.csv'):
        df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
        df = df.asfreq('M') # monthly frequency
        model = ExponentialSmoothing(df['Price'], trend='add', seasonal='add', seasonal_periods=12).fit()
        return model, df

    def estimate_price(date_str, model):
        date = pd.to_datetime(date_str)
        steps = (date.to_period('M') - model.data.dates[-1].to_period('M')).n
        forecast = model.forecast(steps=steps)
        return float(forecast.iloc[-1])

    if __name__ == '__main__':
        model, df = load_and_fit()
        print(estimate_price('2025-09-30', model)) # 1 year out
