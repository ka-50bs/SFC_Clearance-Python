# This is a sample Python script.

import numpy as np
import miepython
import sklearn
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth

from scipy import optimize
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm

class SFC_MieFitting(object):

    def __init__(self):
        '''
        Инициализация инструментальных характеристик цитомтера для решения обратной задачи.

        - λ - длина волны индикатрисного лазера, нм
        - n0 - показаель преломнения обжимающей жидкости (физраствор, вода), у.е.
        - n1 - показатель преломления материала капилляра (кварц), у.е.
        - R - радиус сферического зеркала, м
        - h - высота каплляра, м
        - H - расстояние между сферическим зеркалом до фотоприемником, м
        - d - диаметр капилляра, м
        - Δ - диаметр отверстия диафрагмы, м
        - angles - массив, угловой диапазон вычисления теории Ми, градусы
        - μ - массив, фазовый диапазон вычисления теории Ми, у.е.

        - bounds - кортеж, диапазоны параметров для решения обратной задачи
            * bounds[0] - радиус частицы, нм
            * bounds[1] - показатель преломления частицы, у.е.
            * bounds[2] - скорость частицы (потока обжимающей жидкости), м/с
            * bounds[3] - расстояние до триггера, м
            * bounds[4] - коэффициент Ми/мВ
        '''

        self.n0 = 1.333
        self.n1 = 1.458
        self.λ = 660 / self.n0
        self.R = 2.5 * 10 ** -3
        self.h = 8 * 10 ** -3
        self.H = 180 * 10 ** -3
        self.d = 0.254 * 10 ** -3
        self.Δ = 1.1 * 10 ** -3
        self.fADC = 750000
        self.angles = np.linspace(4, 110, 180)
        self.tf = self.transfer_function(self.angles)

        self.hf = self.hardware_function_m(self.angles) * self.hardware_function(self.angles)
        #self.hf = self.hardware_function_m(self.angles) * self.tf_hf(self.angles)[1]

        self.hf = self.hf / np.amax(self.hf)

        self.μ = np.cos(self.angles / 180 * np.pi)

        self.bounds = ((2.3, 2.6), (-0.0045, -0.0035), (10 ** 4, 10 ** 6))

        self.Iexp = None
        self.Texp = None

    def tf_hf(self, θ):
        '''
        Функция, определяющая передаточную функцию цитометра и аппаратную функицию цитометра.
        Передаточная функция – это функция, определяющая зависимость положения
        частицы l от угла рассеяния θ.
        Аппаратная функция – это функция, преобразующая амплитуду рассеянного света
        в амплитуду сигнала на фотоприемнике в зависимости от угла рассеяния θ.
        В данной
        :param θ: массив, состоящий из углов рассеяния
        :return: возвращает зависимость l(θ)
        '''
        n0 = self.n0
        n1 = self.n1
        R = self.R
        h = self.h
        H = self.H
        d = self.d
        Δ = self.Δ
        θ_rad = θ / (180 / np.pi)

        bt = np.arccos(n0 * np.cos(θ_rad) / n1)
        gm = R * np.sin(bt / 2) / (n1 * H + h * (1 - n1) - R * 3 / 2 * np.cos(bt / 2))
        dx = d / 2 * (1 / np.tan(θ_rad) - 1 / np.tan(bt))
        x = (R * (1 - np.sin(bt / 2 - gm / 2) / np.sin(bt)) + dx)
        a = (1 + (-x + dx) / R) * np.sin(bt)
        phi = -2 * np.arccos(a) - bt + np.pi
        btd = 1.333 / n1 * np.sin(θ_rad) / np.sin(bt)
        dxd = -d / 2 * (1 / (np.sin(θ_rad)) ** 2 - btd / (np.sin(bt)) ** 2)
        dal = -1 / (2 / np.sqrt(1 - a ** 2) * (dxd / R * np.sin(bt) + np.cos(bt) * btd * a / np.sin(bt)) - btd)
        dgm = Δ / (n1 * H + h * (1 - n1) - R * np.cos(bt / 2))
        dal = dal * dgm
        return -x, dal * (180 / np.pi)

    def transfer_function(self, θ):
        '''
        Функция, определяющая передаточную функцию цитометра.
        Передаточная функция – это функция, определяющая зависимость положения
        частицы l от угла рассеяния θ.
        :param θ: массив, состоящий из углов рассеяния
        :return: возвращает зависимость l(θ)
        '''

        n0 = self.n0
        n1 = self.n1
        R = self.R
        h = self.h
        H = self.H
        d = self.d

        θ_rad = θ / (180 / np.pi)
        β = np.arccos(n0 / n1 * np.cos(θ_rad))
        γ = R / (H * n1 + h * (1 - n1)) * (2 * np.sin(β / 2) / (2 - np.cos(β / 2)))
        l = -R * (1 - np.sin(β / 2 - γ / 2) / np.sin(β)) - d / 2 * (1 / np.tan(θ_rad) - 1 / np.tan(β))
        return l

    def hardware_function(self, θ):
        '''
        Функция, определяющая аппаратную функцию цитометра.
        Аппаратная функция – это функция, преобразующая амплитуду рассеянного света
        в амплитуду сигнала на фотоприемнике в зависимости от угла рассеяния θ.
        :param θ: массив, состоящий из углов рассеяния
        :return: возвращает зависимость δθ(θ)
        '''

        n0 = self.n0
        n1 = self.n1
        R = self.R
        h = self.h
        H = self.H
        d = self.d
        Δ = self.Δ

        θ_rad = θ / (180 / np.pi)
        β = np.arccos(n0 / n1 * np.cos(θ_rad))

        γ = R / (H * n1 + h * (1 - n1)) * (2 * np.sin(β / 2) / (2 - np.cos(β / 2)))

        l = -R * (1 - np.sin(β / 2 - γ / 2) / np.sin(β)) - d / 2 * (1 / np.tan(θ_rad) - 1 / np.tan(β))
        a = 1 + l / R
        βθ = np.sin(θ_rad) / np.sin(β)
        dLθ = -d / 2 * (1 / np.sin(θ_rad) ** 2 + βθ / np.sin(β) ** 2)
        δγ = Δ / (H * n1 + h * (1 - n1))
        δθ = -(2 / np.sqrt(1 - a ** 2) * ((dLθ / R) * np.sin(β) + a * βθ * np.cos(β)) - βθ) ** (-1) * δγ
        return δθ

    def hardware_function_m(self, θ):
        '''
        Функция вычисляющая М-функцию
        :param θ: массив, состоящий из углов рассеяния
        :return: возвращает зависимость δθ(θ)
        '''
        m = np.exp(-2 * (np.log(θ / 54)) ** 2) / θ
        return m

    def mie_scat(self, r, n):
        '''
        Функция, описывающая сигнал светарассеяния однородной сферической частицы.
        Для вычисления используется теория Ми.

        :param r: параметр, радиус частицы, указанный в метрах (нм)
        :param n: безразмерный параметр, показатель преломления частицы
        :return: индикатриса светорассеяния, вычисленная по заданным углам I(θ)
        '''
        I = np.array(miepython.ez_intensities(n / self.n0, r, self.λ, self.μ)[0])
        return I

    def trace_zero_norm(self, Ind_exp):
        '''
        Функция удаления постоянной составляющей у трейса I(t).
        Метод вычисления: сортировка значений I(t) по возрастанию, выбор первых 1000 отсортированных значений
        и вычисление для этих значений mean() и std(). Итоговый массив вычисляется как
        I'(t) = I(t) - mean() - std()
        :param Ind_exp: Массив, трейс светорассеяния I(t)
        :return: Массив, обработанный трейс светорассеяния I'(t)
        '''
        Ind_edited = np.copy(Ind_exp)
        for i in range(len(Ind_edited)):
            Ind_edited[i] = Ind_exp[i] - np.mean(np.sort(Ind_exp[i])[:1000]) - 3 * np.std(np.sort(Ind_exp[i])[:1000])
        return Ind_edited

    def gauss_fit(self, x_func, y_func):

        def func(x_, a, x0, sigma):
            return a * np.exp(-(x_ - x0) ** 2 / (2 * sigma ** 2))
        '''
        Фунция подгонки сигнала гауссовским колоколом.
        :param x_func: Массив арументов подгоняемой функции
        :param y_func: Массив значений подгоняемой функции
        :return: Массив найденых параметров A, x0, sigma
        '''
        bounds = [[0, np.amin(x_func), 0],
                  [np.amax(y_func), np.amax(x_func), np.mean(x_func)]]
        popt, pcov = curve_fit(f=func, xdata=x_func, ydata=y_func, bounds=bounds)
        return popt

    def time_cutter(self, Ind_exp, Trig_exp):
        '''
        Функция автоматического определения положение трейса светорассеяния частицы
        и вычленения его из экспериментального сигнала I(t).
        Вычленение производится путем подгонки трейса гауусовской
        функцией и вычислением средних значений mean & std.
        Временное окно вычисляется как (mean - 3 * std, mean + 3 * std)

        :param Ind_exp: Массив, трейс светорассеяния I(t)
        :param Trig_exp: Массив, триггерный сигнал светорассеяния I(t)
        :return: Массив, обработанный трейс светорассеяния I'(t)
        '''

        Ind_exp = np.copy(Ind_exp)[:, :2000]
        Trig_exp = np.copy(Trig_exp)[:, :400]

        time = np.array(range(len(Ind_exp[0])))
        ind_pars = []
        time_array = []

        for i in tqdm(range(len(Ind_exp)),  desc ='Time Cutter Processing'):
            ind_pars.append(self.gauss_fit(time, Ind_exp[i]))

        ind_pars = np.array(ind_pars)
        center = np.mean(ind_pars[:, 1])
        width = np.mean(ind_pars[:, 2])

        for i in tqdm(range(len(Ind_exp)),  desc ='Time Normalizing Processing'):
            itime = (time[int(center - 2 * width): int(center + 2 * width)])
            itime = itime - np.argmax(Trig_exp[i])
            itime = itime / self.fADC
            time_array.append(itime)

        time_array = np.array(time_array)

        return Ind_exp[:, int(center - 2 * width): int(center + 2 * width)], time_array


    def cluster_filter(self, x, y):
        '''
        Функция фильтрации трейсов по производной характеристике сигнала - полному интегралу трейса Sum(I(t)).
        Выборка Sum(I(t)) кластеризуется методом MeanShift, а далее выбирается средний кластер. Это делается в
        в предположении, что сигналы светорассеяния условно делятся на три кластера:
        - Кластер частиц с низким сигналом светорассения (условно мусорный сигнал)
        - Кластер частиц со средним сигналом светорассения (нужный сигнал)
        - Кластер частиц со высоким сигналом светорассения (сигналы с зашкалом)

        :param x:
        :param y:
        :return:
        '''

        x_edited = np.copy(x)
        y_edited = np.copy(y)

        sum_list = []
        for i in range(len(x_edited)):
            sum_list.append(sum(x_edited[i]))

        sum_list = np.reshape(sum_list, (-1, 1))
        bandwidth = estimate_bandwidth(sum_list, quantile=0.1)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(sum_list)
        cluster_center = ms.cluster_centers_
        labels = ms.labels_

        return x_edited[labels == 1], y_edited[labels == 1]

    def trace_th(self, args):
        '''
        Функция, вычисляющая теоритический трейс светорассеяния
        :param args:
        :param Iexp:
        :return:
        '''

        _r, _n, _v, _x0, _α = args[0], args[1], args[2], args[3], args[4]
        mie = self.mie_scat(_r, _n)
        mie = mie * self.hf
        time_th = (self.tf - _x0) / _v
        mie_inter = np.interp(x=self.Texp, xp=time_th, fp=mie, left=0, right=0)
        return mie_inter * _α

    def metric(self, Iexp, Ith):
        def l2_metric(Iexp, Ith):
            return np.sum((Iexp - Ith) ** 2) / np.sum(Iexp)

        def c_mass_metric(Iexp, Ith):
            def c_mass(x):
                return np.sum(x * self.Texp) / np.sum(x)

            return np.abs(c_mass(Iexp) - c_mass(Ith)) / c_mass(Iexp)

        def fft_metric(Iexp, Ith):
            return np.sum((np.fft.hfft(Iexp) - np.fft.hfft(Ith) ** 2)) / np.sum((np.fft.hfft(Iexp)) ** 2)

        return l2_metric(Iexp, Ith)

    def fit(self, Texp, Iexp):
        self.Iexp = Iexp
        self.Texp = Texp

        def mininisation_func(args):
            ss = [3700, 1.584, args[0], args[1], args[2]]
            trace_th = self.trace_th(ss)
            return self.metric(self.Iexp, trace_th)

        res = optimize.direct(func=mininisation_func,
                              bounds=self.bounds,
                              maxfun=10000,
                              maxiter=10000,
                              locally_biased=False,
                              eps=1)

        trace_th = self.trace_th([3700, 1.584, res.x[0], res.x[1], res.x[2]])

        solve = list(res.x)
        return solve, trace_th
    pass





