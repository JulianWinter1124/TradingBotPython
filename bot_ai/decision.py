import math

import numpy as np
from keras.losses import mean_squared_error

from bot import simulation as sim


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


def decide_action_on_prediction(pred, tanh_risk=0.5):
    """

    :param original_data:
    :param pred:
    :param label_indices:
    :param tanh_risk: The sigmoid distance to the win margin. Values should be from tanh(0.5) to <1, where close to 1 is harder to meet but less risky
    :return:
    """
    action = ('hold', 0, 0) #sell, buy, hold
    current_pred = pred[0, :]
    i, j = 0, 1
    last_sign = 2
    while j < len(current_pred):
        p1 = current_pred[i]
        p2 = current_pred[j]
        sign = np.sign(p2-p1)
        if sign == 0:
            j += 1
            continue
        if last_sign != 2 and last_sign != sign:
            break #Better option in the future
        last_sign = sign
        win_margin_price = sim.calc_win_margin_price(p1, sign)
        calculated_tanh_risk = calc_tanh_diff(abs(p2 - p1), abs(win_margin_price - p1)) #test this pls
        if sign == 1:
            if calculated_tanh_risk >= tanh_risk:
                amount = 0 # What to insert here?
                action = ('buy', amount, calculated_tanh_risk)
                break
            else:
                j += 1
        elif sign == -1:
            if calculated_tanh_risk >= tanh_risk:
                amount = 0
                action = ('sell', amount, calculated_tanh_risk)
                break
            else:
                j += 1
    return action

def calc_tanh_diff(margin, min_margin):
    return np.tanh(margin/(np.float64(2.0)*min_margin))
#