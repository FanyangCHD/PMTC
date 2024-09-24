# -*-coding:utf-8-*-

import numpy as np
import math
from scipy import signal

def compare_SNR(recov_img, real_img):

    real = np.linalg.norm(real_img, ord='fro')

    noise = np.linalg.norm(real_img - recov_img, ord='fro')

    if noise == 0 or real==0:
      s = 999.99
    else:
      s = 10*math.log(real/noise, 10)
    return s

def batch_snr(de_data, clean_data):

    De_data = de_data.data.cpu().numpy()  
    Clean_data = clean_data.data.cpu().numpy()
    SNR = 0
    for i in range(De_data.shape[0]):
        De = De_data[i, :, :, :].squeeze()  
        Clean = Clean_data[i, :, :, :].squeeze()
        SNR += compare_SNR(De, Clean)
    return SNR / De_data.shape[0]


def error1(y_pred,y_true):

    # y_true = np.array(y_true)
    # y_pred = np.array(y_pred)
    diff = y_true - y_pred
    num = np.linalg.norm(diff, ord=2)
    den = np.linalg.norm(y_true, ord=2)
    error = num / den
    return error

def mse1(img1, img2):
    return np.mean((img1 - img2) ** 2)

def calculate_error(y_pred,y_true):

    y_pred = y_pred.data.cpu().numpy() 
    y_true = y_true.data.cpu().numpy() 
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    num1 = np.sum((y_true - y_pred) ** 2)
    num = np.sqrt(num1)
    den1 = np.sum(y_true ** 2)
    den = np.sqrt(den1)
    return num / den

def r_squared(y_pred, y_true):

    y_pred = y_pred.data.cpu().numpy() 
    y_true = y_true.data.cpu().numpy() 
    mean_true = sum(y_true) / len(y_true)
    total_sum_squares = sum((y - mean_true) ** 2 for y in y_true)
    residual_sum_squares = sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true)))
    
    r_squared = 1 - (residual_sum_squares / total_sum_squares)
    
    return r_squared

def calculate_rmse(y_pred,y_true):

    y_pred = y_pred.data.cpu().numpy() 
    y_true = y_true.data.cpu().numpy() 
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    mse1 = math.sqrt(mse)
    return mse1
