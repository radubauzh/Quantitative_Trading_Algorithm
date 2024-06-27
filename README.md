# Data gathering and pre processing

The File also includes all programs for the data gathering. Main file is the Jupiter Notebook!

The needed libraries are:

- yfinance
- pandas
- numpy
- csv
- openpyxl
- matplotlib.pyplot
- tensorflow
- keras
- sklearn
- seaborn
- scikit-learn

Run the following command to install the needed libraries:

`pip install numpy pandas matplotlib seaborn keras tensorflow scikit-learn`

The code is divided into three main parts:

1. Data gathering `run.py` -> creates our used CSV, which served as datafoundation
2. Hyperparametrization File _Hyperparametrization_ -> each executable to calculate best hyper parameter for the models. -> returns in shell command line the parameters and visuals.
3. Best Model calculation and evaluation _jupiter File_ -> return None. All visuals used can be recalculated within here.

The main file is the SP500_Trading_Algrithm file. It contains all necessary information regarding the model creation and the model evaluation. The file is divided into the following parts:

1. Feature Engineering: Data Preprocessing, Feature Selection, Feature Engineering. 
2. Model Creation: LSTM, CNN, FFNN, Autoencoder
3. Model Evaluation: 
## Authors

- [Rafael Dubach](https://github.com/radubauzh)
- [David Diener](https://github.com/Dave5252)
- [Felix Wallhorn](https://github.com/FWALL9)
