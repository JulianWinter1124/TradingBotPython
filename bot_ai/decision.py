import numpy as np
from keras.losses import mean_squared_error


def make_decision_of_predicition(pred, original_data, close_column_index, risk=0.1):
    #pred.shape = (len, n_out+1)
    #1. look how good predicitons have been
    n_pred = pred.shape[0]
    n_columns = pred.shape[1]
    original_selection = original_data[-n_pred-n_columns:, close_column_index]
    cols = list()
    for i in range(n_columns):
        cols.append(np.roll(original_selection, i, axis=0))
    original_selection = np.hstack(cols)[-n_pred:, :] #original_selection.shape == pred.shape
    mse = mean_squared_error(original_selection[:-n_columns, :], pred[:-n_columns, :]) #compare only data that is all known
    #?
    #compare mse tp safety somehow
    calc_risk = mse * 1.0 #Make meaningful calculation (Range between 0 and 1) probably min max
    if calc_risk <= risk: #Buy/Sell
        #Buy/ Sell amount based on calculated certainity and pred
        #Maybe Test with Monte-Carlo
        current_price = original_data[-1, close_column_index]
        pass
    else:
        print("Risk threshold not met")
