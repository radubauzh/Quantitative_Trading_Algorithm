# Prsentation

## I. Introduction (Felix)

- Early start in semester, no precise instructions for project
- Initial idea: Start with data gathering via APIs and if possible APP
- After precise instruction: Use procvided Excel
- Get s&p data from yfinance and adjust to excel

## II. Experiments Intro (Felix)
### 1 Experiment
- Hypo: All features bring the best indication
- Deatils: Utilized 30 indicators, combining macroeconomic indicators, market sentiment factors, and commodity prices.
### 2 Experiment
- Hypo: Macro features bring the best indication
- Details:Focused on five macroeconomic indicators: GDP, CPI, unemployment rate, nominal interest rate, and 10-year treasury yield.
- Capture broad market trends
### 3 Experiment
- Hypo: Market features bring the best indication
- Details: Used five market and technical indicators: price/earnings ratio, dividend yield, market returns, value index, and growth index.
- Reflect the actual strenth of the economy 
### Trading strategy
- Each week we decide to go long or short
- Predict via the difference from the next prediction 
- Evaluate via the last prediction (if we were correct)


## III. NETWORK ARCHITECTURES (Rafael)
 
A. Feedforward Neural Networks (FFNN)
- Simple architecture
B. Autoencoders
- Encoder, decoder
C. Convolutional Neural Networks (CNN)
- 1D
D. Recurrent Neural Networks (RNN) and Long Short-Term Memory Networks (LSTM)
- Sequential data

- Overall takes to prevent overfitting: 
    - Early stop
    - Small learning rate 
    - Drop out
    - Dataset shuffling 
    - No Data augmentaition :synthetic manipulations might introduce patterns that do not exist in the real world, noise might be unbeneficial

- Hyperparameter tuning
    - Droput rate, batch size, learning rate, optimizer, Units, epochs, filter amounts, kernel size (CNN)
-  Normaliztion
    - Approapreate normalization

## Evaluation of Experiments 
### 1 Experiment (Rafael)
- Hypo: All features bring the best indication
- Results: 
    - CNN demonstrated the best predictive performance with the lowest mean squared error (MSE) and highest R2 values.
    - Sharpe Ratio: LSTM had the highest Sharpe Ratio (1.125), indicating superior risk-adjusted returns.

 **Show actual vs. fitted**

### 2 Experiment (David)
- Hypo: Macro features bring the best indication
- Results: 
    - LSTM outperformed other models with the lowest MSE and highest R2 values.
    - Sharpe Ratio: LSTM again showed the highest Sharpe Ratio (1.121), indicating strong predictive capability.

 **Show residual plots, actual vs. fitted**

### 3 Experiment (David)
- Hypo: Market features bring the best indication
- Results:  
    - Both CNN and FFNN showed strong performance in predicting based on market and technical indicators.
    - Sharpe Ratio: LSTM exhibited the highest Sharpe Ratio (1.120), indicating robust risk-adjusted returns.
 **Show residual plots, actual vs. fitted**


## Overall Conclusion (David)
- Different indicator sets and network architectures can effectively predict S&P 500 movements.
- CNN and LSTM models showed robust predictive performance and reliable trading signals.
- Key Takeaway: A multi-faceted approach leveraging diverse data sources enhances the predictive power of neural network models in financial markets.