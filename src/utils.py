import time
import numpy as np
class process_bar(object):
    def __init__(self, total_tik_number, std_scale = 3.0, move_sacle = 0.9):
        self.total_tik_number = total_tik_number
        self.std_scale = std_scale
        self.move_sacle = move_sacle

    def start(self):
        self.tik_list = [time.time()]
        self.delta_tik_list = []
        self.move_mean = None
    
    def _calc_mean_time(self):
        time_std = np.std(self.delta_tik_list[-10:]) * self.std_scale
        if self.move_mean is None:
            predict_move_mean = self.delta_tik_list[0]
        else:
            predict_move_mean = self.move_mean*self.move_sacle + self.delta_tik_list[-1]*(1-self.move_sacle)
        if abs(self.delta_tik_list[-1]-predict_move_mean)>time_std:
            self.move_mean = self.delta_tik_list[-1]
        else:
            self.move_mean = predict_move_mean

    def tik(self):
        self.tik_list.append(time.time())
        self.delta_tik_list.append(self.tik_list[-1]-self.tik_list[-2])
        self._calc_mean_time()

        process_past = len(self.delta_tik_list) / self.total_tik_number
        time_total = self.move_mean * self.total_tik_number
        time_left = time_total * (1-process_past)
        return process_past, time_total, time_left

def float2time(f_time):
    _f_time = f_time
    str_time = ''
    if f_time>=3600*24:
        day = _f_time // (3600*24)
        _f_time -= day * 3600 * 24
        str_time += '%dd '%day
    if f_time>=3600:
        h = _f_time // 3600
        _f_time -= h * 3600
        str_time += '%dh '%h
    m = _f_time // 60
    _f_time -= m * 60
    str_time += '%dm '%m
    str_time += '%ds'%_f_time
    return str_time
