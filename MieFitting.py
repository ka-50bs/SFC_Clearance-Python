# This is a sample Python script.

import numpy as np
import miepython
import sklearn
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import MinMaxScaler

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
            * bounds[0] - d, размер частицы, нм
            * bounds[1] - n, показатель преломления частицы, у.е.
            * bounds[2] - v, скорость частицы (потока обжимающей жидкости), м/с
            * bounds[3] - l0, расстояние до триггера, м
            * bounds[4] - I, коэффициент мВ/Ми
        '''

        self.n0 = 1.333333333
        self.n1 = 1.458
        self.λ = 660
        self.R = 2.5 * 10 ** -3
        self.h = 8 * 10 ** -3
        self.H = 180 * 10 ** -3
        self.d = 0.254 * 10 ** -3
        self.Δ = 1.1 * 10 ** -3
        self.fADC = 750000
        self.angles = np.linspace(10, 110, 180)


        self.μ = np.cos(self.angles / 180 * np.pi)

        self.bounds_vxa = ((2.3, 2.6),
                           (-0.0045, -0.0035),
                           (10 ** 5, 3 * 10 ** 5))

        self.bounds_dnvxa = ((3600, 3800),
                             (1.57, 1.59),
                             (2.3, 2.6),
                             (-0.0045, -0.0035),
                             (10 ** 5, 3 * 10 ** 5))

        self.bounds_dnv = ((3400, 4000),
                           (1.50, 1.60),
                           (2.3, 2.6))

        self.Iexp = None
        self.Texp = None

    def tf_hf_init(self):
        self.tf = self.transfer_function(self.angles)
        self.hf = self.hardware_function_m(self.angles)
        self.hf = self.hf / np.amax(self.hf)

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
        :return: возвращает зависимость MF(θ)
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
        I = np.array(miepython.ez_intensities(n, r, self.λ, self.μ, n_env = self.n0)[0])
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
        try:
            popt, pcov = curve_fit(f=func, xdata=x_func, ydata=y_func, bounds=bounds)
        except:
            print('Bad Signal!')
            popt = np.zeros(3)
        # plt.plot(y_func, '*')
        # plt.plot(func(x_func, popt[0],popt[1],popt[2]))
        # plt.show()
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

        def func(x_, a, x0, sigma):
            return a * np.exp(-(x_ - x0) ** 2 / (2 * sigma ** 2))

        Ind_exp = np.copy(Ind_exp)[:, :2000]
        Trig_exp = np.copy(Trig_exp)[:, :400]

        time = np.array(range(len(Ind_exp[0])))
        time_trig = np.array(range(len(Trig_exp[0])))

        bounds = [[0, np.amin(time_trig), 0],
                  [np.amax(Trig_exp), np.amax(time_trig), np.mean(time_trig)]]

        ind_pars = []
        time_array = []

        for i in tqdm(range(len(Ind_exp)),  desc ='Time Cutter Processing'):
            ind_pars.append(self.gauss_fit(time, Ind_exp[i]))

        ind_pars = np.array(ind_pars)
        center = np.mean(ind_pars[:, 1])
        width = np.mean(ind_pars[:, 2])

        for i in tqdm(range(len(Ind_exp)),  desc ='Time Normalizing Processing'):
            itime = (time[int(center - 1 * width): int(center + 1 * width)])
            itime = itime - np.argmax(Trig_exp[i])
            itime = itime / self.fADC
            time_array.append(itime)

        time_array = np.array(time_array)

        return Ind_exp[:, int(center - 1 * width): int(center + 1 * width)], time_array


    def cluster_int_filter(self, x, y):
        '''
        Функция фильтрации трейсов по производной характеристике сигнала - полному интегралу трейса Sum(I(t)).
        Выборка Sum(I(t)) кластеризуется методом MeanShift. На данном этапе производится обычная разметка
        данных без предварительного выбора нужного кластера частиц. Для того, чтобы выбрать нужный кластер,
        нужно вне функции выбрать требуемый тип частиц (кластер). Возможно, через несколько иттераций будет добавлено
        выбор кластера по стороннему критерию.
        :param x: Входной массив предобработанных трейсов, по которыс производится классификация
        :param y: Входной массив временных точек трейсов. Не используется в функции, но может понадобится при выборе
        нужного кластера частиц, если ,будет сделана модификация функции

        :return: x_edited: выходной массив трейсов
        :return: y_edited: выходной масиив временных точек трейсов
        :return: labels: выходной масиив
        '''

        x_edited = np.copy(x)
        y_edited = np.copy(y)

        sum_list = []
        for i in range(len(x_edited)):
            sum_list.append(sum(x_edited[i]))

        sum_list = np.reshape(sum_list, (-1, 1))
        bandwidth = estimate_bandwidth(sum_list, quantile=0.3)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(sum_list)
        cluster_center = ms.cluster_centers_
        labels = ms.labels_

        return x_edited, y_edited, labels

    def cluster_cm_filter(self, x, y):
        '''

        :return: x_edited: выходной массив трейсов
        :return: y_edited: выходной масиив временных точек трейсов
        :return: labels: выходной масиив
        '''

        x_edited = np.copy(x)
        y_edited = np.copy(y)

        cm_list = []
        for i in range(len(x_edited)):
            y_cm = np.sum(x_edited[i] * y_edited[i]) / np.sum(x_edited[i])
            x_cm = np.sum(x_edited[i] * y_edited[i]) / np.sum(y_edited[i])
            cm_list.append((y_cm, x_cm))

        cm_list = np.array(cm_list)

        scaler = MinMaxScaler()
        cm_list = scaler.fit_transform(cm_list)

        bandwidth = estimate_bandwidth(cm_list, quantile=0.3)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(cm_list)
        cluster_center = ms.cluster_centers_
        labels = ms.labels_

        for i in range(len(np.unique(labels))):
            data = cm_list[labels == i]
            plt.plot(data[:,0],data[:,1], '*', label='label%d' %(i))

        plt.legend()
        plt.show()

        return x_edited, y_edited, labels

    def trace_th(self, args):
        '''
        Функция, вычисляющая теоритический трейс светорассеяния
        :param args: [r, _n, _v, _x0, _α]
        :return:
        '''

        _r, _n, _v, _x0, _α = args[0], args[1], args[2], args[3], args[4]
        mie = self.mie_scat(_r, _n)
        mie = mie * self.hf
        time_th = (self.tf - _x0) / _v
        mie_inter = np.interp(x=self.Texp, xp=time_th, fp=mie, left=0, right=0)
        return mie_inter * _α

    def metric(self, Iexp, Ith):
        '''
        Функция метрики различия двух сигналов,
        :param Iexp: Массив экспериментального сигнала
        :param Ith: Массив теоретического сигнала
        :return: Значение нужной метрики (L2)
        '''
        def l2_metric(Iexp, Ith):
            return np.sum((Iexp - Ith) ** 2)# / np.sum(Iexp ** 2)
        return l2_metric(Iexp, Ith)


    def bayesian_errors(self, mse_map, mse_pars, I_exp):
        '''
        Функция вычисления ошибок методом Байесса.

        :param mse_map:
        :param mse_pars:
        :param I_exp:
        :return:
        '''

        N = len(I_exp)

        auto_corr = np.zeros(N)
        for tau in range(N):
            for t in range(N):
                if t - tau > 0:
                    auto_corr[tau] = auto_corr[tau] + I_exp[t] * I_exp[t - tau]

        n = N ** 2 / (N + 2 * sum(N - k for k in range(1, N - 1)) * auto_corr)
        S = lambda x: x ** (-n / 2)
        k = (np.sum(S(mse_map))) ** -1
        mean = k * np.sum(mse_pars * S, axis = 0)
        sigma = k * np.sum((mean - mse_pars) * S, axis = 0)
        return mean, sigma

    def fit_serial(self, Texp_array, Iexp_array):
        '''
        Функция, решающая обратную задачу светорассеяния для сферической однороднной латексной частицы описанной пятью
        параметрами методом псоследовательного фиттинга.
        :param Texp_array:
        :param Iexp_array:
        :return:
        '''

        def fit_vli(Texp, Iexp):
            '''
            Функция, решающая обратную задачу трех параметров для латексной частицы:
            * v, скорость частицы (потока обжимающей жидкости), м/с
            * l0, расстояние до триггера, м
            * I0, коэффициент мВ/Ми
            Решение производится в предположении о том, что известны параметры размера и показателя преломления
            исследуемой сферической однородной частицы, а именно d = 3700nm, n = 1.584
            :param Texp:
            :param Iexp:
            :return:
            '''

            self.Iexp = Iexp
            self.Texp = Texp

            def mininisation_func_vli(args):
                params = [3700, 1.584, args[0], args[1], args[2]]
                Ith = self.trace_th(params)
                return self.metric(Iexp, Ith)

            res = optimize.direct(func=mininisation_func_vli,
                                  bounds=self.bounds_vxa,
                                  maxfun=5000,
                                  maxiter=5000,
                                  locally_biased=False,
                                  eps=1)
            solve = list([3700, 1.584, res.x[0], res.x[1], res.x[2]])
            #trace_th = self.trace_th(solve)
            return solve

        def fit_dnv(Texp, Iexp, l0, I0):
            '''
            Функция, решающая обратную задачу трех параметров для латексной частицы:
            * d, размер частицы, нм
            * n, показатель преломления частицы, у.е.
            * v, скорость частицы (потока обжимающей жидкости), м/с
            Решение производится в предположении о том, что известны параметры расстояния до триггера и коффициента мв/Ми
            исследуемой сферической однородной частицы, а именно d = 3700nm, n = 1.584
            :param Texp:
            :param Iexp:
            :param l0:
            :param I0:
            :return:
            '''

            self.Iexp = Iexp
            self.Texp = Texp

            def mininisation_func_dnv(args):
                params = [args[0], args[1], args[2], l0, I0]
                Ith = self.trace_th(params)
                return self.metric(Iexp, Ith)

            res = optimize.direct(func=mininisation_func_dnv,
                                  bounds=self.bounds_dnv,
                                  maxfun=5000,
                                  maxiter=5000,
                                  locally_biased=False,
                                  eps=1)
            solve = list([res.x[0], res.x[1], res.x[2], l0, I0])
            #trace_th = self.trace_th(solve)
            return solve



        N_array = len(Texp_array)

        pre_solve = []
        solve = []

        for i in tqdm(range(N_array), desc ="Pre_Solvation"):
            pre_solve.append(fit_vli(Texp_array[i], Iexp_array[i]))

        pre_solve = np.array(pre_solve)

        (global_lo, global_Io) = (np.mean(pre_solve[:, 3]), np.mean(pre_solve[:, 4]))

        for i in tqdm(range(N_array), desc ="Solvation"):
            solve.append(fit_dnv(Texp_array[i], Iexp_array[i], global_lo, global_Io))

        solve = np.array(solve)
        return solve

    def fit_global(self, Texp, Iexp):
        '''
        Функция, решающая обратную задачу светорассеяния для сферической однороднной латексной частицы описанной пятью
        параметрами методом глобального фиттинга.
        :param Texp:
        :param Iexp:
        :return:
        '''

        self.Iexp = Iexp
        self.Texp = Texp

        def mininisation_func_rnvxa(args):
            params = [args[0], args[1], args[2], args[3], args[4]]
            Ith = self.trace_th(params)
            return self.metric(Iexp, Ith)

        res = optimize.direct(func=mininisation_func_rnvxa,
                              bounds=self.bounds_dnvxa,
                              maxfun=5000,
                              maxiter=5000,
                              locally_biased=False,
                              eps=1)
        solve = list([res.x[0], res.x[1], res.x[2], res.x[3], res.x[4]])
        #trace_th = self.trace_th(solve)
        return solve

    pass





