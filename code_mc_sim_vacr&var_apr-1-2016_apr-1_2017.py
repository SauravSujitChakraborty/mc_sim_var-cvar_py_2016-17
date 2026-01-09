import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from yahooquery import Ticker

# --- Function Definitions (remain unchanged) ---

def get_data(stocks, start, end):
    # Custom User-Agent to help avoid the 429 "Too Many Requests" error
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    
    # Initialize Ticker with the user_agent
    t = Ticker(stocks, user_agent=user_agent)
    stock_data = t.history(start=start, end=end)
    
    # Handle the case where the request might still fail due to rate limiting
    if stock_data is None or isinstance(stock_data, dict):
        print("Error: Could not fetch data. You might still be rate-limited by Yahoo.")
        return None, None

    # Pivot data and clean missing values
    stockData = stock_data['close'].unstack(level=0)
    returns = stockData.pct_change().dropna()
    
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    
    return meanReturns, covMatrix

def mcVaR(returns, alpha=5):
    """
    Input: pandas series of returns
    Output: percentile on return distribution to a given confidence level alpha
    """
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    else:
        raise TypeError("Expected a pandas data series.")

def mcCVaR(returns, alpha=5):
    """
    Input: pandas series of returns
    Output: CVaR or Expected Shortfall to a given confidence level alpha
    """
    if isinstance(returns, pd.Series):
        belowVaR = returns <= mcVaR(returns, alpha=alpha)
        return returns[belowVaR].mean()
    else:
        raise TypeError("Expected a pandas data series.")

# --- Main Simulation Logic ---

stockList = ['CBA', 'BHP', 'TLS', 'NAB', 'WBC', 'STO']
stocks = [stock + '.AX' for stock in stockList]

# Set the desired date range: Apr 1 2017 to Apr 1 2018
startDate = dt.datetime(2016, 4, 1) 
endDate = dt.datetime(2017, 4, 1)

# Fetch data
meanReturns, covMatrix = get_data(stocks, startDate, endDate)

# Only proceed if data was successfully retrieved
if meanReturns is not None:
    # Generate random weights
    weights = np.random.random(len(meanReturns))
    weights /= np.sum(weights)

    mc_sims = 100 
    T = 365
    initialPortfolio = 10000

    # Prepare matrices for calculations
    meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns.values)
    meanM = meanM.T
    portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

    # Run simulations
    for m in range(0, mc_sims):
        Z = np.random.normal(size=(T, len(weights)))
        # Added noise to ensure matrix is positive definite for Cholesky
        L = np.linalg.cholesky(covMatrix + np.eye(len(weights)) * 1e-8)
        dailyReturns = meanM + np.inner(L, Z)
        portfolio_sims[:,m] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*initialPortfolio

    # --- Calculations & Results ---
    
    # Extract the results from the last day of simulation
    portResults = pd.Series(portfolio_sims[-1,:])

    VaR = initialPortfolio - mcVaR(portResults, alpha=5)
    CVaR = initialPortfolio - mcCVaR(portResults, alpha=5)

    print('VaR ${}'.format(round(VaR, 2)))
    print('CVaR ${}'.format(round(CVaR, 2)))

    # Plot the results
    plt.figure(figsize=(10,6))
    plt.plot(portfolio_sims)
    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Days')
    plt.title('MC simulation of a stock portfolio (Apr 1 2016 - Apr 1 2017)')
    plt.show()

