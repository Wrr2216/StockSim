StockSim

StockSim is a Python project that uses historical stock data to predict whether to buy or sell a given stock. It uses various technical indicators such as moving averages, relative strength index (RSI), and moving average convergence divergence (MACD) to make its predictions.
Getting Started

To use StockSim, you will need to have Python 3 installed on your computer. You will also need to install several Python packages, which can be done by running the following command:

pip install -r requirements.txt

Next, you can run the main.py file and enter the stock symbol you wish to analyze when prompted. This will download historical stock data from Yahoo Finance and use it to train a support vector machine (SVM) model to make predictions on future price changes.
Technical Indicators

StockSim uses the following technical indicators to make its predictions:

    Simple Moving Average (SMA) - calculates the average price of a stock over a certain number of time periods
    Relative Strength Index (RSI) - measures the strength of a stock's price action and whether it is overbought or oversold
    Moving Average Convergence Divergence (MACD) - measures the difference between two moving averages and can be used to identify trend reversals

Machine Learning Model

StockSim uses a support vector machine (SVM) model to make predictions on future price changes. The SVM is trained on historical data using the technical indicators mentioned above, as well as whether the stock price increased or decreased the following day.
Conclusion

StockSim is a simple Python project that uses technical analysis and machine learning to predict whether to buy or sell a given stock. While the predictions made by StockSim should not be relied upon as financial advice, they can provide a starting point for further research and analysis.