#import data_gathering_labeling
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Beispiel für eine Liste von Data-Objekten (data_list)
# data_list = [Data(emp=..., pe=..., ...), Data(emp=..., pe=..., ...), ...]

class DataPoint:
    def __init__(self, emp=None, pe=None, cape=None, dy=None, rho=None, mov=None, ir=None, rr=None, y02=None, y10=None, stp=None, cf=None, mg=None, rv=None, ed=None, un=None, gdp=None, m2=None, cpi=None, dil=None, yss=None, nyf=None, _au=None, _dxy=None, _lcp=None, _ty=None, _oil=None, _mkt=None, _va=None, _gr=None, snp=None, label=None):
        self.emp = emp
        self.pe = pe
        self.cape = cape
        self.dy = dy
        self.rho = rho
        self.mov = mov
        self.ir = ir
        self.rr = rr
        self.y02 = y02
        self.y10 = y10
        self.stp = stp
        self.cf = cf
        self.mg = mg
        self.rv = rv
        self.ed = ed
        self.un = un
        self.gdp = gdp
        self.m2 = m2
        self.cpi = cpi
        self.dil = dil
        self.yss = yss
        self.nyf = nyf
        self._au = _au
        self._dxy = _dxy
        self._lcp = _lcp
        self._ty = _ty
        self._oil = _oil
        self._mkt = _mkt
        self._va = _va
        self._gr = _gr
        self.snp = snp
        self.label = label

    def __repr__(self) -> str:
        return f"{self.emp}, {self._oil}, {self.stp}, {self.label}\n"

def all_data_no_refinement(self):
    # Anzahl der Datenpunkte und Features
    n_samples = len(self.emp)
    n_features = 31  # 30 features plus SNP

    # Initialisiere Arrays für Features (X) und Labels (y)
    X = np.zeros((n_samples, n_features + 1))
    y = np.zeros(n_samples)

    data_list = []
    for emp, pe, cape, dy, rho, mov, ir, rr, y02, y10, stp, cf, mg, rv, ed, un, gdp, m2, cpi, dil, yss, nyf, _au, _dxy, _lcp, _ty, _oil, _mkt, _va, _gr, snp, label in zip(self.emp, self.pe, self.cape, self.dy, self.rho, self.mov, self.ir, self.rr, self.y02, self.y10, self.stp, self.cf, self.mg, self.rv, self.ed, self.un, self.gdp, self.m2, self.cpi, self.dil, self.yss, self.nyf, self._au, self._dxy, self._lcp, self._ty, self._oil, self._mkt, self._va, self._gr, self.snp, self.label):
        one_datapoint = DataPoint(emp=emp.get_value(), pe=pe.get_value(), cape=cape.get_value(), dy=dy.get_value(), rho=rho.get_value(), mov=mov.get_value(), ir=ir.get_value(), rr=rr.get_value(), y02=y02.get_value(), y10=y10.get_value(), stp=stp.get_value(), cf=cf.get_value(), mg=mg.get_value(), rv=rv.get_value(), ed=ed.get_value(), un=un.get_value(), gdp=gdp.get_value(), m2=m2.get_value(), cpi=cpi.get_value(), dil=dil.get_value(), yss=yss.get_value(), nyf=nyf.get_value(), _au=_au.get_value(), _dxy=_dxy.get_value(), _lcp=_lcp.get_value(), _ty=_ty.get_value(), _oil=_oil.get_value(), _mkt=_mkt.get_value(), _va=_va.get_value(), _gr=_gr.get_value(),snp=snp.get_value(), label=label)
        data_list.append(one_datapoint)
        

    for i, data_point in enumerate(data_list):
        X[i, 0] = data_point.emp
        X[i, 1] = data_point.pe
        X[i, 2] = data_point.cape
        X[i, 3] = data_point.dy
        X[i, 4] = data_point.rho
        X[i, 5] = data_point.mov
        X[i, 6] = data_point.ir
        X[i, 7] = data_point.rr
        X[i, 8] = data_point.y02
        X[i, 9] = data_point.y10
        X[i, 10] = data_point.stp
        X[i, 11] = data_point.cf
        X[i, 12] = data_point.mg
        X[i, 13] = data_point.rv
        X[i, 14] = data_point.ed
        X[i, 15] = data_point.un
        X[i, 16] = data_point.gdp
        X[i, 17] = data_point.m2
        X[i, 18] = data_point.cpi
        X[i, 19] = data_point.dil
        X[i, 20] = data_point.yss
        X[i, 21] = data_point.nyf
        X[i, 22] = data_point._au
        X[i, 23] = data_point._dxy
        X[i, 24] = data_point._lcp
        X[i, 25] = data_point._ty
        X[i, 26] = data_point._oil
        X[i, 27] = data_point._mkt
        X[i, 28] = data_point._va
        X[i, 29] = data_point._gr
        X[i, 30] = data_point.snp
        y[i] = data_point.label

    #print("NaN values in X:", np.isnan(X).sum())
    #print("NaN values in y:", np.isnan(y).sum())

    # Überprüfe die Form der extrahierten Daten
    print(f"Shape von X: {X.shape}")
    print(f"Shape von y: {y.shape}")

    selected_features = ["emp", "pe", "cape", "dy", "rho", "mov", "ir", "rr", "y02", "y10", "stp", "cf", "mg", "rv", "ed", "un", "gdp", "m2", "cpi", "dil", "yss", "nyf", "_au", "_dxy", "_lcp", "_ty", "_oil", "_mkt", "_va", "_gr", "snp", "label"]

    return selected_features, X, y

def correlation(self):
    # Anzahl der Datenpunkte und Features
    n_samples = len(self.emp)
    n_features = 31  # 30 features plus SNP

    # Initialisiere Arrays für Features (X) und Labels (y)
    X = np.zeros((n_samples, n_features + 1))
    y = np.zeros(n_samples)

    data_list = []
    for emp, pe, cape, dy, rho, mov, ir, rr, y02, y10, stp, cf, mg, rv, ed, un, gdp, m2, cpi, dil, yss, nyf, _au, _dxy, _lcp, _ty, _oil, _mkt, _va, _gr, snp, label in zip(self.emp, self.pe, self.cape, self.dy, self.rho, self.mov, self.ir, self.rr, self.y02, self.y10, self.stp, self.cf, self.mg, self.rv, self.ed, self.un, self.gdp, self.m2, self.cpi, self.dil, self.yss, self.nyf, self._au, self._dxy, self._lcp, self._ty, self._oil, self._mkt, self._va, self._gr, self.snp, self.label):
        one_datapoint = DataPoint(emp=emp.get_value(), pe=pe.get_value(), cape=cape.get_value(), dy=dy.get_value(), rho=rho.get_value(), mov=mov.get_value(), ir=ir.get_value(), rr=rr.get_value(), y02=y02.get_value(), y10=y10.get_value(), stp=stp.get_value(), cf=cf.get_value(), mg=mg.get_value(), rv=rv.get_value(), ed=ed.get_value(), un=un.get_value(), gdp=gdp.get_value(), m2=m2.get_value(), cpi=cpi.get_value(), dil=dil.get_value(), yss=yss.get_value(), nyf=nyf.get_value(), _au=_au.get_value(), _dxy=_dxy.get_value(), _lcp=_lcp.get_value(), _ty=_ty.get_value(), _oil=_oil.get_value(), _mkt=_mkt.get_value(), _va=_va.get_value(), _gr=_gr.get_value(),snp=snp.get_value(), label=label)
        data_list.append(one_datapoint)
        

    for i, data_point in enumerate(data_list):
        X[i, 0] = data_point.emp
        X[i, 1] = data_point.pe
        X[i, 2] = data_point.cape
        X[i, 3] = data_point.dy
        X[i, 4] = data_point.rho
        X[i, 5] = data_point.mov
        X[i, 6] = data_point.ir
        X[i, 7] = data_point.rr
        X[i, 8] = data_point.y02
        X[i, 9] = data_point.y10
        X[i, 10] = data_point.stp
        X[i, 11] = data_point.cf
        X[i, 12] = data_point.mg
        X[i, 13] = data_point.rv
        X[i, 14] = data_point.ed
        X[i, 15] = data_point.un
        X[i, 16] = data_point.gdp
        X[i, 17] = data_point.m2
        X[i, 18] = data_point.cpi
        X[i, 19] = data_point.dil
        X[i, 20] = data_point.yss
        X[i, 21] = data_point.nyf
        X[i, 22] = data_point._au
        X[i, 23] = data_point._dxy
        X[i, 24] = data_point._lcp
        X[i, 25] = data_point._ty
        X[i, 26] = data_point._oil
        X[i, 27] = data_point._mkt
        X[i, 28] = data_point._va
        X[i, 29] = data_point._gr
        X[i, 30] = data_point.snp
        y[i] = data_point.label

    print("NaN values in X:", np.isnan(X).sum())
    print("NaN values in y:", np.isnan(y).sum())

    # Überprüfe die Form der extrahierten Daten
    print(f"Shape von X: {X.shape}")
    print(f"Shape von y: {y.shape}")

    # Create a DataFrame from data_list
    df = pd.DataFrame([vars(data_point) for data_point in data_list])

    # Compute pairwise correlation of columns, excluding NA/null values
    correlation_matrix = df.corr()

    # Optional: Visualize the correlation matrix

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()

    # Assuming 'label' is the name of your target variable column
    correlations_with_target = correlation_matrix['label'].drop('label')  # Exclude target's self-correlation
    print("Correlation with target variable:\n", correlations_with_target)

    # Example: Select features highly correlated with target (absolute correlation coefficient >= 0.7)
    important_features = correlations_with_target[abs(correlations_with_target) >= 0.7].index.tolist()
    print("Important features based on correlation with target:\n", important_features)

    # List of selected feature names (adjust based on your analysis)
    selected_features = important_features

    # Update number of features
    n_features = len(selected_features)

    # Initialize X with the selected features
    X = np.zeros((n_samples, n_features + 1))  # Add 1 for SNP if needed

    # Map selected features to their corresponding indices in X
    feature_indices = {feature: idx for idx, feature in enumerate(selected_features)}
    #print(feature_indices)

    # Populate X with selected feature values
    for i, data_point in enumerate(data_list):
        for feature_name in selected_features:
            X[i, feature_indices[feature_name]] = getattr(data_point, feature_name)

        # Assuming SNP is another feature you want to include
        # X[i, n_features] = data_point.snp

        # Populate label y
        y[i] = data_point.label

    #print(y)

    print("Shape of X after feature selection:", X.shape)
    print("Shape of y:", y.shape)

    print("NaN values in X:", np.isnan(X).sum())
    print("NaN values in y:", np.isnan(y).sum())

    return selected_features, X, y