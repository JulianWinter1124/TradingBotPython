import math

import numpy as np

from bot import simulation as sim, API


# def make_decision_of_predicition(pred, original_data, close_column_index, risk=0.1):
#     #pred.shape = (len, n_out+1)
#     #1. look how good predicitons have been
#     n_pred = pred.shape[0]
#     n_columns = pred.shape[1]
#     original_selection = original_data[-n_pred-n_columns:, close_column_index]
#     cols = list()
#     for i in range(n_columns):
#         cols.append(np.roll(original_selection, i, axis=0))
#     original_selection = np.hstack(cols)[-n_pred:, :] #original_selection.shape == pred.shape
#     mse = mean_squared_error(original_selection[:-n_columns, :], pred[:-n_columns, :]) #compare only data that is all known
#     #?
#     #compare mse tp safety somehow
#     calc_risk = mse * 1.0 #Make meaningful calculation (Range between 0 and 1) probably min max
#     if calc_risk <= risk: #Buy/Sell
#         #Buy/ Sell amount based on calculated certainity and pred
#         #Maybe Test with Monte-Carlo
#         current_price = original_data[-1, close_column_index]
#         pass
#     else:
#         print("Risk threshold not met")


def decide_action_on_prediction(pair, pred, state, current_price, exclude_current_price=True, tanh_risk=0.5):
    """
    Decides on a given predicition what actions to take. This function uses a tanh function to evaluate the distance between loss and win.
    This method does NOT evaluate how likely the predicition is.
    :param pair: The crypto pair that is being evaluated
    :param pred: The predicition (considered true)
    :param state: the simulation or real state
    :param tanh_risk: The sigmoid distance to the win margin. Values should be in range tanh(0.5) < tanh_risk < 1, where close to 1 is harder to meet but less risky.
    :return: The action to take in the format: (buy/sell/hold, amount, calculated_tanh_risk)
    """
    action = (pair, 'hold', 0, -1) #action = (pair, {sell, buy, hold}, amount, stop-loss)
    current_pred = pred[0, :]
    if not exclude_current_price:
        np.insert(current_pred, 0, current_price)
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
            print('better option in the future')
            break #Better option in the future
        last_sign = sign
        win_margin_price = sim.calc_win_margin_price(p1, sign)
        calculated_tanh_risk = calc_tanh_diff(abs(p2 - p1), abs(win_margin_price - p1))
        if sign == 1:
            if calculated_tanh_risk >= tanh_risk:
                max_loss = current_price*0.01
                amount = state.get_dollar_balance*0.02/max_loss
                stop_loss = current_price-max_loss
                action = (pair, 'buy', amount, stop_loss)
                break
            else:
                j += 1
        elif sign == -1:
            if calculated_tanh_risk >= tanh_risk:
                cur = state.extract_currency_to_buy_from_pair(pair)
                amount = state.get_curreny_balance(cur)*0.98 #sell 98%
                action = (pair, 'sell', amount, -1)
                break
            else:
                j += 1
    return action

def calc_tanh_diff(margin, min_margin):
    return np.tanh(margin/(np.float64(2.0)*min_margin))
#