import os
import data_gathering_labeling as dgl
import attribute_reduction as ar
import model1

if __name__ == '__main__':
    # Foundation

    path = "data/market_data.xlsx"
    mainData = dgl.Data()
    dgl.read_data(self=mainData, path=path)
    dgl.request_snp(self=mainData)
    dgl.labeling_data(self=mainData)
    #dgl.create_csv(self=mainData)
    

    # Feature refinement


    # getting all features back
    priority_features = ar.all_data_no_refinement(self=mainData)

    # getting only features correlated to 
    #priority_features = ar.correlation(self=mainData)


    # Neural nets

    # attempt for convolutional net
    #model1.prediction_model(self=mainData, selected_features=priority_features)
    
    # Time series model -> currently overfitted
    #model1.time_series_analysis(self=mainData)

    #model1.your_prediciton_model(self=mainData, features = priority_features)