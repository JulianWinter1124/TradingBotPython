import math
from random import randint

import numpy as np

from bot import simulation as sim, API


def decide_action_on_prediction(pair, pred, state, current_price, exclude_current_price=True, tanh_risk=0.5):
    """
    Decides on a given predicition what actions to take. This function uses a tanh function to evaluate the distance between loss and win.
    This method does NOT evaluate how likely the predicition is.
    :param current_price: The current price for the pair
    :param exclude_current_price: if the current price should be ignored and only act within prediction
    :param pair: The crypto pair that is being evaluated
    :param pred: The predicition (considered true)
    :param state: the simulation or real state
    :param tanh_risk: tanh distance to the win margin. Values should be in range tanh(0.5) < tanh_risk < 1, where close to 1 is harder to meet but less risky.
    :return: The action to take in the format: (pair, buy/sell/hold, amount, stop-loss(if used))
    """
    action = (pair, 'hold', 0, None)  # action = (pair, {sell, buy, hold}, amount, stop-loss)
    current_pred = pred[0, :] #prediction is of shape [[x1, x2, x3, x4]]
    if not exclude_current_price:
        current_pred = np.insert(current_pred, 0, current_price) #Add current price at front if not ignored
    i, j = 0, 1
    last_sign = 2
    while j < len(current_pred):
        p1 = current_pred[i]
        p2 = current_pred[j]
        sign = np.sign(p2 - p1) #determine if up or down trend
        if sign == 0:
            j += 1 # if no trend move second pointer one to the right
            continue
        if last_sign != 2 and last_sign != sign: # if trends switch in between there will be a better time to act.
            #print('better option in the future')
            break
        last_sign = sign
        win_margin_price = state.calc_win_margin_price(p1, sign) #calculated the minimum needed price to make profit
        calculated_tanh_risk = calc_tanh_diff(abs(p2 - p1), abs(win_margin_price - p1)) #Check if distance is big enough with tangens function
        if sign == 1: #up trend
            if calculated_tanh_risk >= tanh_risk: #this means buy
                max_loss = current_price * 0.05 #The maximum amount you are willing to lose
                amount = (state.get_dollar_balance() * 0.002) / max_loss #the amount to invest. formula from https://www.investopedia.com/terms/p/positionsizing.asp
                stop_loss = current_price - max_loss #put stop loss here
                action = (pair, 'buy', amount, stop_loss) #put together action
                break
            else:
                j += 1
        elif sign == -1:
            if calculated_tanh_risk >= tanh_risk: #This means sell
                cur = state.extract_first_currency_from_pair(pair)
                amount = state.get_currency_balance(cur) * 0.998  # sell 98%
                action = (pair, 'sell', amount, None)
                break
            else:
                j += 1
    return action

def make_random_action(pair, state, current_price):
    action = (pair, 'hold', 0, None)
    r = randint(0, 2)
    if r == 0:
        max_loss = current_price * 0.05  # The maximum amount you are willing to lose
        amount = (state.get_dollar_balance() * 0.002) / max_loss  # the amount to invest. formula from https://www.investopedia.com/terms/p/positionsizing.asp
        stop_loss = current_price - max_loss  # put stop loss here
        action = (pair, 'buy', amount, stop_loss)  # put together action
    elif r == 1:
        cur = state.extract_first_currency_from_pair(pair)
        amount = state.get_currency_balance(cur) * 0.998  # sell 98%
        action = (pair, 'sell', amount, None)
    elif r == 2:
        pass #hold
    return action

def calc_tanh_diff(distance, min_distance):
    """
    If the distance from the current price is big enough this function's value is > tanh(1/2)
    the further away the closer to 1 this function gets
    :param distance: the current distance from the current price
    :param min_distance: the needed minimum distance from the current price
    :return:
    """
    return np.tanh(distance / (np.float64(2.0) * min_distance))


def stringify_action(action): #converts the action to a printable (not pretty) string
    pair, actionstr, amount, stop_loss = action
    if actionstr == 'buy':
        return 'buy '+str(amount)+ pair + ' and place stop-loss at '+ str(stop_loss)+ '$'
    elif actionstr == 'sell':
        return 'sell '+ str(amount)+' of '+ pair
    elif actionstr == 'hold':
        return 'hold ' + pair + ' and do nothing'
    if action is None:
        return 'action not understood'
#
