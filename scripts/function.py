import os
import random
import bisect
import torch
import cv2
import numpy as np
import tifffile
import glob


def set_seed(seed=666):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sum_hist(hist):
    a = np.zeros([256, 1], dtype=np.float32)
    pre_hist = np.float32(0)
    for i in range(256):
        a[i, 0] = pre_hist+hist[i, 0]
        pre_hist = a[i, 0]
    a = a/pre_hist
    return a


def img_preprocess(sumhist_s, img):
    exist_value = np.unique(img)
    hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    sumhist = sum_hist(hist)
    table = np.zeros([256, 1], dtype=np.uint8)
    for k in exist_value:
        pos = bisect.bisect_left(sumhist_s, sumhist[k])
        if pos == 0:
            table[k] = np.uint8(0)
        else:
            pos_pre = np.where(sumhist_s == sumhist_s[pos-1])[0][0]
            table[k] = np.uint8((sumhist[k]-sumhist_s[pos_pre])/(sumhist_s[pos]-sumhist_s[pos_pre])*(pos-pos_pre)+pos_pre)
    img = cv2.LUT(img, table)
    return img


def image_cut_extend(I, m, n, c1, c2, t):
    C = np.zeros((c1+2*t, c2+2*t), dtype=np.float32)
    C[t:t+c1, t:t+c2] = I[m*c1:(m+1)*c1, n*c2:(n+1)*c2]
    if m == 0:# top
        C[0:t, t:c2+t] = np.flip(I[m*c1:m*c1+t, n*c2:(n+1)*c2], axis=0)
    else:
        C[0:t, t:c2+t] = I[m*c1-t:m*c1, n*c2:(n+1)*c2]
    if m == (I.shape[0]/c1-1):# bottom
        C[c1+t:c1+2*t, t:c2+t] = np.flip(I[(m+1)*c1-t:(m+1)*c1, n*c2:(n+1)*c2], axis=0)
    else:
        C[c1+t:c1+2*t, t:c2+t] = I[(m+1)*c1:(m+1)*c1+t, n*c2:(n+1)*c2]
    if n == 0:# left
        C[t: c1+t, 0:t] = np.flip(I[m*c1:(m+1)*c1, n*c2:n*c2+t], axis=1)
    else:
        C[t: c1+t, 0:t] = I[m*c1:(m+1)*c1, n*c2-t:n*c2]
    if n == (I.shape[1]/c2-1):# right
        C[t:c1+t, c2+t:c2+2*t] = np.flip(I[m*c1:(m+1)*c1, (n+1)*c2-t:(n+1)*c2], axis=1)
    else:
        C[t:c1+t, c2+t:c2+2*t] = I[m*c1:(m+1)*c1, (n+1)*c2:(n+1)*c2+t]
    if m == 0 or n == 0:# top-left
        C[0:t, 0:t] = np.rot90(I[m*c1:m*c1+t, n*c2:n*c2+t], 2)
    else:
        C[0:t, 0:t] = I[m*c1-t:m*c1, n*c2-t:n*c2]
    if m == (I.shape[0]/c1-1) or n == 0:# bottom-left
        C[c1+t:c1+2*t, 0:t] = np.rot90(I[(m+1)*c1-t:(m+1)*c1, n*c2:n*c2+t], 2)
    else:
        C[c1+t:c1+2*t, 0:t] = I[(m+1)*c1:(m+1)*c1+t, n*c2-t:n*c2]
    if m == 0 or n == (I.shape[1]/c2-1):# top-right
        C[0:t, c2+t:c2+2*t] = np.rot90(I[m*c1:m*c1+t, (n+1)*c2-t:(n+1)*c2], 2)
    else:
        C[0:t, c2+t:c2+2*t] = I[m*c1-t:m*c1, (n+1)*c2:(n+1)*c2+t]
    if m == (I.shape[0]/c1-1) or n == (I.shape[1]/c2-1):# bottom-right
        C[c1+t:c1+2*t, c2+t:c2+2*t] = np.rot90(I[(m+1)*c1-t:(m+1)*c1, (n+1)*c2-t:(n+1)*c2], 2)
    else:
        C[c1+t:c1+2*t, c2+t:c2+2*t] = I[(m+1)*c1:(m+1)*c1+t, (n+1)*c2:(n+1)*c2+t]
    return C


def sum_hist_16(hist):
    sum_hist = np.zeros([65536, 1], dtype=np.float32)
    pre_hist = np.float32(0)
    for i in range(65536):
        sum_hist[i, 0] = pre_hist+hist[i, 0]
        pre_hist = sum_hist[i, 0]
    sum_hist = sum_hist/pre_hist
    return sum_hist


def take_close_16(a, number):
    pos = bisect.bisect_left(a, number)
    if pos == 0:
        return np.uint16(0)
    else:
        pos_pre = np.where(a == a[pos-1])[0][0]
        value = np.uint16((number-a[pos_pre])/(a[pos]-a[pos_pre])*(pos-pos_pre)+pos_pre)
        return value


def img_preprocess_16(image_s, image, threshold_s=None, threshold=None, select_range=None):
    img_s = image_s.copy()
    img = image.copy()
    exist_value = np.unique(img)
    table = np.zeros([65536, 1], dtype=np.uint16)
    if threshold_s is None and threshold is None:
        if select_range is not None:
            mask_s = img_s.copy()
            min = np.percentile(img_s, select_range[0])
            max = np.percentile(img_s, select_range[1])
            mask_s[mask_s <= min] = 0
            mask_s[mask_s >= max] = 0
            mask_s[mask_s != 0] = 255
            mask_s = np.uint8(mask_s)
        else:
            mask_s = None
        hist_s = cv2.calcHist([img_s], [0], mask_s, [65536], [0, 65535])
        sumhist_s = sum_hist_16(hist_s)
        hist = cv2.calcHist([img], [0], None, [65536], [0, 65535])
        sumhist = sum_hist_16(hist)
        for k in exist_value:
            table[k] = take_close_16(sumhist_s, sumhist[k])
    else:
        mask_s1 = img_s.copy()
        mask_s2 = img_s.copy()
        if select_range is not None:
            min = np.percentile(img_s, select_range[0])
            max = np.percentile(img_s, select_range[1])
            mask_s1[mask_s1 >= max] = 0
            mask_s1[mask_s1 <= threshold_s] = 0
            mask_s1[mask_s1 != 0] = 255
            mask_s1 = np.uint8(mask_s1)
            mask_s2[mask_s2 <= min] = 0
            mask_s2[mask_s2 > threshold_s] = 0
            mask_s2[mask_s2 != 0] = 255
            mask_s2 = np.uint8(mask_s2)
        else:
            mask_s1[mask_s1 <= threshold_s] = 0
            mask_s1[mask_s1 > threshold_s] = 255
            mask_s1 = np.uint8(mask_s1)
            mask_s2 = 255 - mask_s1
        mask1 = img.copy()
        mask1[mask1 <= threshold] = 0
        mask1[mask1 > threshold] = 255
        mask1 = np.uint8(mask1)
        mask2 = 255 - mask1
        hist_s1 = cv2.calcHist([img_s], [0], mask_s1, [65536], [0, 65535])
        sumhist_s1 = sum_hist_16(hist_s1)[threshold_s+1:]
        hist1 = cv2.calcHist([img], [0], mask1, [65536], [0, 65535])
        sumhist1 = sum_hist_16(hist1)
        hist_s2 = cv2.calcHist([img_s], [0], mask_s2, [65536], [0, 65535])
        sumhist_s2 = sum_hist_16(hist_s2)[:threshold_s+1]
        hist2 = cv2.calcHist([img], [0], mask2, [65536], [0, 65535])
        sumhist2 = sum_hist_16(hist2)
        for k in exist_value:
            if k > threshold:
                table[k] = take_close_16(sumhist_s1, sumhist1[k])+threshold_s+1
            else:
                table[k] = take_close_16(sumhist_s2, sumhist2[k])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = table[img[i, j]]
    return img
