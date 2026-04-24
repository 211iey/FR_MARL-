import pandas as pd
import numpy as np
import yfinance as yf

# 1. Load data
excess = pd.read_csv('data/excess_returns(2012 start).csv', parse_dates=['Date'], index_col='Date')
log_ret = pd.read_csv('data/etf_log_Returns(2012 start).csv', parse_dates=['Date'], index_col='Date')

# 2. Load VKOSPI
vkospi = pd.read_csv('data/vkospi.csv', encoding='utf-8')
# Clean date column (has spaces like "2026- 04- 03")
vkospi.columns = vkospi.columns.str.strip().str.replace('"', '')
vkospi['날짜'] = vkospi['날짜'].str.replace('"', '').str.replace(' ', '')
vkospi['날짜'] = pd.to_datetime(vkospi['날짜'])
vkospi['종가'] = vkospi['종가'].astype(str).str.replace('"', '').str.replace(',', '').astype(float)
vkospi = vkospi.set_index('날짜').sort_index()
vkospi = vkospi[['종가']].rename(columns={'종가': 'vkospi'})

# 3. Download KOSPI index from yfinance
print("Downloading KOSPI index data...")
kospi = yf.download('^KS11', start='2011-12-01', end='2026-04-10', auto_adjust=True)
kospi_close = kospi['Close'].squeeze()
kospi_log_ret = np.log(kospi_close / kospi_close.shift(1)).dropna()
kospi_log_ret.name = 'kospi_log_ret'

# 4. Calculate vol20 / vol60 for each stock
stock_cols = log_ret.columns.tolist()

vol20_df = log_ret[stock_cols].rolling(window=20).std()
vol20_df.columns = [f'vol20_{c}' for c in stock_cols]

vol60_df = log_ret[stock_cols].rolling(window=60).std()
vol60_df.columns = [f'vol60_{c}' for c in stock_cols]

# 5. Calculate vol20(kospi)
kospi_vol20 = kospi_log_ret.rolling(window=20).std()
kospi_vol20.name = 'vol20_kospi'
# Flatten multi-index if present
if hasattr(kospi_vol20.index, 'levels'):
    kospi_vol20.index = kospi_vol20.index.get_level_values(0)
kospi_vol20.index = pd.to_datetime(kospi_vol20.index).tz_localize(None)

# 6. Merge everything onto excess_returns
result = excess.copy()
result = result.join(vkospi, how='left')
result = result.join(vol20_df, how='left')
result = result.join(vol60_df, how='left')
result = result.join(kospi_vol20, how='left')

# 7. Save
result.to_csv('data/excess_returns(2012 start).csv')
print(f"Done! Shape: {result.shape}")
print(f"Columns: {result.columns.tolist()}")
print(f"Date range: {result.index.min()} ~ {result.index.max()}")
print(f"\nSample (first non-NaN vol20 row):")
first_valid = result['vol20_semicon'].first_valid_index()
print(result.loc[first_valid, ['vkospi', 'vol20_semicon', 'vol60_semicon', 'vol20_kospi']])
