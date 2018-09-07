import math

import numpy as np

from bot import simulation as sim, API


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
    action = (pair, 'hold', 0, None)  # action = (pair, {sell, buy, hold}, amount, stop-loss)
    current_pred = pred[0, :]
    if not exclude_current_price:
        current_pred = np.insert(current_pred, 0, current_price)
    i, j = 0, 1
    last_sign = 2
    while j < len(current_pred):
        p1 = current_pred[i]
        p2 = current_pred[j]
        sign = np.sign(p2 - p1)
        if sign == 0:
            j += 1
            continue
        if last_sign != 2 and last_sign != sign:
            #print('better option in the future')
            break  # Better option in the future
        last_sign = sign
        win_margin_price = sim.calc_win_margin_price(p1, sign)
        calculated_tanh_risk = calc_tanh_diff(abs(p2 - p1), abs(win_margin_price - p1))
        if sign == 1:
            if calculated_tanh_risk >= tanh_risk:
                max_loss = current_price * 0.1
                amount = (state.get_dollar_balance() * 0.02) / max_loss
                stop_loss = current_price - max_loss
                action = (pair, 'buy', amount, stop_loss)
                break
            else:
                j += 1
        elif sign == -1:
            if calculated_tanh_risk >= tanh_risk:
                cur = state.extract_currency_to_buy_from_pair(pair)
                amount = state.get_currency_balance(cur) * 0.98  # sell 98%
                action = (pair, 'sell', amount, None)
                break
            else:
                j += 1
    return action


def calc_tanh_diff(distance, min_distance):
    """
    If the distance from the current price is big enough this function's value is > tanh(1/2)
    :param distance: the current distance from the current price
    :param min_distance: the needed minimum distance from the current price
    :return:
    """
    return np.tanh(distance / (np.float64(2.0) * min_distance))


def stringify_action(action):
    pair, actionstr, amount, stop_loss = action
    if actionstr == 'buy':
        return 'buy '+ pair+' for '+str(amount)+'$ and place stop-loss at '+ str(stop_loss)+ '$'
    elif actionstr == 'sell':
        return 'sell '+ str(amount)+' of '+ pair
    elif actionstr == 'hold':
        return 'hold ' + pair + ' and do nothing'
    if action is None:
        return 'action not understood'
#
