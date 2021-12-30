# %%
# import modules
%pylab inline 
import os
import pandas as pd
import numpy as np
import pandas_datareader as web
from datetime import datetime
import random as random
import seaborn as sns

# %%
# load stocks
symbols = ["AMD","AMZN","FB","PETS","CRM"]
start = datetime(2019,1,1)
end = datetime(2021,11,30)
stock_data = web.get_data_yahoo(symbols,start, end)
daily_adj_close=stock_data["Adj Close"]

# %%
# plot closing price
daily_adj_close.plot()
plt.title("Target Stock Adjusted Closing Price")
plt.ylabel("Ajusted Closing Price Over Time")
plt.xlabel("Date")

# %%
# calculate daily simple rate
plt.figure(figsize=(30,10))
daily_simple_return = daily_adj_close.pct_change()
daily_simple_return.plot()
plt.title("Daily Simple Return")
plt.ylabel("ROR")
plt.xlabel("Date")
print(daily_simple_return.head())


# %%
# create subplots
plt.figure(figsize = (16,16))
plt.subplot(3,2,1)
daily_simple_return.AMD.plot()
plt.title("AMD")
plt.subplot(3,2,2)
daily_simple_return.AMZN.plot()
plt.title("Amazon")
plt.subplot(3,2,3)
daily_simple_return.FB.plot()
plt.title("Facebook")
plt.subplot(3,2,4)
daily_simple_return.PETS.plot()
plt.title("PETS")
plt.subplot(3,2,5)
daily_simple_return.CRM.plot()
plt.title("Salesforce")
plt.subplots_adjust(wspace=0.2, hspace=0.5)

# %%
# calculate mean of daily simple rate of return
mean_daily_amd = daily_simple_return.AMD.mean()
mean_daily_amazon = daily_simple_return.AMZN.mean()
mean_daily_facebook = daily_simple_return.FB.mean()
mean_daily_pets = daily_simple_return.PETS.mean()
mean_daily_salesforce = daily_simple_return.CRM.mean()
mean_portforlio_daily_simple_return = [mean_daily_amd,mean_daily_amazon,mean_daily_facebook,mean_daily_pets,mean_daily_salesforce]
plt.bar (range(len(symbols)),mean_portforlio_daily_simple_return)
ax = plt.subplot()
plt.title("Mean of Daily Simple Return")
plt.ylabel ("Mean of Return")
plt.xlabel("Stock Portfolio")
ax.set_xticks(range(len(symbols)))
ax.set_xticklabels(symbols)


# %%
# calculate variance & standard deviation
variance_return = daily_simple_return.var()
print("Variance of Daily Simple Return: \n", variance_return)
variance_return.plot.bar()
plt.title("Variance of Daily Return")
plt.ylabel("Variance")
plt.xlabel("Stock Portfolio")
plt.show()

standard_dev = daily_simple_return.std()
print("Standard Deviation of Daily Simple Return: \n", standard_dev)
standard_dev.plot.bar()
plt.title("Standard Deviation of Daily Return")
plt.ylabel("Standard Deviation")
plt.xlabel("Stock Portfolio")
plt.show()
# %%
# calculate covariance and correlations
covariance_return = daily_simple_return.cov()
print("Covariance \n", covariance_return)
correlation_return = daily_simple_return.corr()
print("Correlations\n", correlation_return)

corr_heatmap = sns.heatmap(covariance_return, vmin=-1, vmax=1, annot=True, cmap='BrBG')

# %%
# calculate efficient frontier
number_assets = 5
weights = np.random.random(number_assets)
weights /= np.sum(weights)
weights_record = []
portfolio_returns = []
portfolio_volatilities = []
sharpe_ratio = []
for single_portfolio in range (10000):
      weights = np.random.random(number_assets)
      weights /= np.sum(weights) 
      returns = np.dot(weights, mean_portforlio_daily_simple_return)
      volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_return, weights)))
      portfolio_returns.append(returns)
      portfolio_volatilities.append(volatility)
      sharpe = returns / volatility
      sharpe_ratio.append(sharpe)
      weights_record.append(weights)
# make returns, volatility and weights into arrays and dataframes 
portfolio_returns = np.array(portfolio_returns)
portfolio_volatilities = np.array(portfolio_volatilities)
sharpe_ratio=np.array(sharpe_ratio)
weights_record_dataframe = pd.DataFrame(weights_record)
weights_record_dataframe.columns= symbols
portforlio_dataframe=  pd.DataFrame({"Returns": portfolio_returns, "Volatilities": portfolio_volatilities, "Sharpe Ratio": sharpe_ratio})
# calculate max sharpe ratio scenario
max_sharpe = sharpe_ratio.argmax()
max_return=portforlio_dataframe.Returns[max_sharpe]
max_volatility =portforlio_dataframe.Volatilities[max_sharpe]
print("Max Sharpe Ratio Scenario:")
print( portforlio_dataframe.iloc[max_sharpe])
print("Weights distribution")
print(weights_record_dataframe.iloc[max_sharpe])

# %%
# calculate min volatility scenario
min_volatility_index = portfolio_volatilities.argmin()
min_volatility_return = portfolio_returns[min_volatility_index]
min_volatility = portfolio_volatilities[min_volatility_index]
print("Min volatility Scenario:")
print(portforlio_dataframe.iloc[min_volatility_index])
print("Weights distribution")
print(weights_record_dataframe.iloc[min_volatility_index])

# %%
# plot efficient frontier
#plt.style.use('seaborn-dark')
plt.figure(figsize=(9, 5))
plt.scatter(portfolio_volatilities, portfolio_returns, c=sharpe_ratio,cmap='RdYlGn', edgecolors='black',marker='o') 
plt.grid(True)
plt.xlabel('expected daily volatility')
plt.ylabel('expected daily return')
plt.title("Efficient Frontier")
plt.colorbar(label='Sharpe ratio')
# plot sharpe ratio max
plt.scatter (max_volatility,max_return,color= "red", marker="*", s = 200)
# plot min volatility
plt.scatter (min_volatility, min_volatility_return, color = 'yellow', marker="*", s= 200)
plt.show()


# %%
