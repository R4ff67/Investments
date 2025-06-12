import numpy as np
import pandas as pd
from scipy import stats
from functools import reduce

# ============================================================================
# Question 6 a)
# ============================================================================


# STEP 1: Construct a new DataFrame that will have all of the 3-month interbank rate corresponding to each country

#Creation of a dictionary with 'key' the name of the currency, and 'value' the csv file of the concerned currency
files = {
    'AUD': 'AUSTRALIA_IB.csv',
    'EUR': 'EUROPE_IB.csv',
    'JPY': 'JAPAN_IB.csv',
    'CHF': 'SWITZERLAND_IB.csv',
    'GBP': 'GREATBRITAIN_IB.csv',
    'USD': 'USA_IB.csv'
}

dfs = [] #Creation of an empty DataFrame where we will append for each column the interbank rate of each currency.

for currency, file in files.items():
    df = pd.read_csv(file)
    interest_col = df.columns[1]  # gets the second column which is the interbank rate
    df = df.rename(columns={'observation_date': 'Date', interest_col: currency})
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=False)  # Convert Date to datetime
    df = df[['Date', currency]]
    dfs.append(df)


# Merge all dataframes on 'Date'
df_rates = reduce(lambda left, right: pd.merge(left, right, on='Date', how='outer'), dfs)

df_rates = df_rates.sort_values('Date').reset_index(drop=True)


#Overview of our new DataFrame
print("IB Rates at the beginning of each month:\n")
print(df_rates.head())



#STEP 2: calculate the carry for each currency from 04/2002 until 12/24 and put them in a DataFrame

# We need to set 'Date' as an index for our next operation or else our code will try to use the substraction operation on it
df_rates = df_rates.set_index('Date')

# Shift dates to end of month
df_rates.index = df_rates.index + pd.offsets.MonthEnd(0)

# Our carry DataFrame consists of substracting USD from each row’s values, and we drop the USD column.
carry_df = df_rates.drop(columns='USD').subtract(df_rates['USD'], axis=0)


# We can now reset the index
carry_df.index.name = 'Date'

# Overview of our carry DataFrame
print("\ncarry = foreign IB - US IB:")
print(carry_df.head())



#STEP 3: compute ranks and weights and store them in a DataFrame

carry_ranks = carry_df.rank(axis=1, method='first')  # computes the rank of each currency for each month and put them in a DataFrame

N = carry_df.shape[1]

center = (N + 1) / 2

raw_scores = carry_ranks - center

# Calculate sums of positives (long) and negatives (short) for each row
long_sums = raw_scores.clip(lower=0).sum(axis=1)   # sum of positive weights per month
short_sums = raw_scores.clip(upper=0).sum(axis=1)  # sum of negative weights per month (negative values)

# Calculate scaling factors to normalize longs to +1 and shorts to -1
Z_long = 1 / long_sums
Z_short = -1 / short_sums  # short_sums is negative, so -1/negative = positive scaling

# Apply scaling factors row-wise to positives and negatives separately
def scale_weights(row):
    longs = row.clip(lower=0) * Z_long[row.name]
    shorts = row.clip(upper=0) * Z_short[row.name]
    return longs + shorts

weights = raw_scores.apply(scale_weights, axis=1)

# Now, each row in weights sums to zero, with longs summing to +1 and shorts summing to -1.
print("\n Weights:")
print(weights.head())

weights.to_csv('weights')



#FINAL STEP: calculate the return for each month and store them in a DataFrame
#We have a file for the excess returns in USD computed in question 3

X_df = pd.read_csv('X_currency_carry.csv', index_col=0)        # First column is the index (dates)
X_df.index = pd.to_datetime(X_df.index)            # Convert index to datetime
X_df = X_df[['AUD', 'EUR', 'JPY', 'CHF', 'GBP']]   # Reorder columns if needed

print("\n Currency hedged index return:")
print(X_df.head())



# Ensure both DataFrames have the same date range
common_dates = weights.index.intersection(X_df.index)
weights_aligned = weights.loc[common_dates]
X_aligned = X_df.loc[common_dates]

# Calculate carry returns: R_CARRY_t+1 = sum(w_i_t * X_i_t+1)
carry_returns = (weights_aligned * X_aligned).sum(axis=1)

# Shift dates forward by 1 month to show actual return month
carry_returns.index = carry_returns.index + pd.offsets.MonthEnd(1)
carry_returns_df = pd.DataFrame(carry_returns, columns=['Carry_Return'])

print("Carry Strategy Returns:")
print(carry_returns_df.head())

# If you want to save the results
carry_returns_df.to_csv('carry_strategy_returns.csv')



# ============================================================================
# Question 6 b) - Analysis of Carry Strategy Components
# ============================================================================

# First, we need to separate the long and short legs of the strategy
# Long leg: currencies with positive weights (high carry)
# Short leg: currencies with negative weights (low carry)

# Calculate long and short leg returns separately
long_weights = weights_aligned.clip(lower=0)  # Keep only positive weights
short_weights = weights_aligned.clip(upper=0)  # Keep only negative weights

# Calculate returns for each leg
long_returns = (long_weights * X_aligned).sum(axis=1)
short_returns = (short_weights * X_aligned).sum(axis=1)

# Shift dates forward by 1 month to show actual return month
long_returns.index = long_returns.index + pd.offsets.MonthEnd(1)
short_returns.index = short_returns.index + pd.offsets.MonthEnd(1)

# We need to recalculate total strategy returns with proper date shifting
total_returns = (weights_aligned * X_aligned).sum(axis=1)
total_returns.index = total_returns.index + pd.offsets.MonthEnd(1)  # Shift dates forward

print("=== CARRY STRATEGY ANALYSIS ===\n")

# Calculate statistics for each component
def calculate_stats(returns, name):
    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe_ratio = mean_ret / std_ret if std_ret != 0 else 0
    
    # Statistical significance test (t-test against zero)
    n = len(returns)
    t_stat = mean_ret / (std_ret / np.sqrt(n)) if std_ret != 0 else 0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))  # Two-tailed test
    
    print(f"{name}:")
    print(f"  Mean Return: {mean_ret:.4f} ({mean_ret*12:.4f} annualized)")
    print(f"  Standard Deviation: {std_ret:.4f} ({std_ret*np.sqrt(12):.4f} annualized)")
    print(f"  Sharpe Ratio: {sharpe_ratio:.4f} ({sharpe_ratio*np.sqrt(12):.4f} annualized)")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Statistically significant at 5%: {'Yes' if p_value < 0.05 else 'No'}")
    print()

# Calculate statistics for all components
calculate_stats(long_returns, "LONG LEG (High Carry Currencies)")
calculate_stats(short_returns, "SHORT LEG (Low Carry Currencies)")
calculate_stats(total_returns, "TOTAL CARRY STRATEGY")





'''print("=== DEBUGGING CARRY STRATEGY ===\n")

# Check the first few rows of weights to understand the signs
print("First few rows of weights:")
print(weights_aligned.head())
print()

# Check first few rows of X_aligned (returns)
print("First few rows of X_aligned (excess returns):")
print(X_aligned.head())
print()

# Let's manually calculate the first row to verify
print("=== MANUAL CALCULATION FOR FIRST ROW ===")
first_date = weights_aligned.index[0]
print(f"Date: {first_date}")
print(f"Weights: {weights_aligned.loc[first_date].values}")
print(f"Returns: {X_aligned.loc[first_date].values}")

# Manual calculation
manual_calc = (weights_aligned.loc[first_date] * X_aligned.loc[first_date]).sum()
print(f"Manual total return: {manual_calc:.6f}")

# Check what our code calculated
auto_calc = (weights_aligned * X_aligned).sum(axis=1).loc[first_date]
print(f"Code calculated: {auto_calc:.6f}")
print()



# Let's see the distribution of weights (positive vs negative)
print("=== WEIGHT DISTRIBUTION ===")
positive_weights = weights_aligned > 0
negative_weights = weights_aligned < 0

print("Average number of positive weights per month:", positive_weights.sum(axis=1).mean())
print("Average number of negative weights per month:", negative_weights.sum(axis=1).mean())
print()

# Check if our long/short separation is correct
print("=== LONG/SHORT SEPARATION CHECK ===")
sample_date = weights_aligned.index[0]
print(f"For {sample_date}:")
print("Original weights:", weights_aligned.loc[sample_date].values)
print("Long weights (clipped lower=0):", weights_aligned.loc[sample_date].clip(lower=0).values)
print("Short weights (clipped upper=0):", weights_aligned.loc[sample_date].clip(upper=0).values)
print()

# The issue might be in how we're interpreting the short leg
# Let's recalculate more carefully
print("=== CORRECTED CALCULATION ===")

# Long leg: positive weights × returns
long_contributions = weights_aligned.clip(lower=0) * X_aligned
long_returns_corrected = long_contributions.sum(axis=1)

# Short leg: negative weights × returns (this should typically give negative contribution)
short_contributions = weights_aligned.clip(upper=0) * X_aligned
short_returns_corrected = short_contributions.sum(axis=1)

print(f"First row long contributions: {long_contributions.iloc[0].values}")
print(f"First row short contributions: {short_contributions.iloc[0].values}")
print(f"First row long return: {long_returns_corrected.iloc[0]:.6f}")
print(f"First row short return: {short_returns_corrected.iloc[0]:.6f}")
print(f"Sum: {(long_returns_corrected.iloc[0] + short_returns_corrected.iloc[0]):.6f}")

# Check means
print(f"\nLong leg mean: {long_returns_corrected.mean():.6f}")
print(f"Short leg mean: {short_returns_corrected.mean():.6f}")
print(f"Total mean: {(long_returns_corrected + short_returns_corrected).mean():.6f}")

# The short leg mean should typically be negative!
# If it's positive, there might be an issue with the data or calculation'''
