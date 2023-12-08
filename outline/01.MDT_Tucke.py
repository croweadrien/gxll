"""
利用MDT将一维数据转为高维数据，提取核心张量后再逆MDT得到最终结果
作者：LIUYANGSHUO
日期：2022年09月22日
"""
import numpy as np
import pandas as pd
from BHT_ARIMA.util.MDT import MDTWrapper
import tensorly as tl
import tensorly as tl
from tensorly.decomposition import tucker


def _initilizer(T_hat, Js, Rs, Xs):
    # initilize Us
    U = [np.random.random([j, r]) for j, r in zip(list(Js), Rs)]
    # initilize es
    # begin_idx = _p + _q
    # es = [[np.random.random(Rs) for _ in range(self._q)] for t in range(begin_idx, T_hat)]
    return U


def _forward_MDT(data, taus):
    mdt = MDTWrapper(data, taus)
    trans_data = mdt.transform()
    _T_hat = mdt.shape()[-1]
    return trans_data, mdt


def _get_Xs(trans_data):
    T_hat = trans_data.shape[-1]
    Xs = [trans_data[..., t] for t in range(T_hat)]

    return Xs


def _get_cores(Xs, Us):
    cores = [tl.tenalg.multi_mode_dot(x, [u.T for u in Us], modes=[i for i in range(len(Us))]) for x in Xs]
    return cores


def _inverse_MDT(mdt, data, taus, shape):
    return mdt.inverse(data, taus, shape)


if __name__ == '__main__':
    #  读取数据
    station = 'ZL'
    tucker_rank = 5

    ori_data_pd = pd.read_csv('./input/{}_0.8_n.csv'.format(station), header=None, index_col=0)
    ori_data_ = ori_data_pd.values

    tucker_rank = ori_data_.shape[-1] - 2
    Rs = [tucker_rank, tucker_rank]  # tucker decomposition ranks

    ts_ = ori_data_[:, :]
    label_ = ori_data_[:, ]

    taus_ = [label_.shape[0], tucker_rank]

    #  MDT变化
    fore_shape = list(ts_.shape)

    trans_data, mdt = _forward_MDT(ts_, taus_)
    core, factors = tucker(trans_data, rank=[33, 5, 5])  # 核心張量、因子矩陣
    new_core = tl.tenalg.multi_mode_dot(core, factors)  # 還原的新張量
    result = _inverse_MDT(mdt, new_core, taus_, fore_shape)
    result_int = np.around(result)
    result = pd.DataFrame(result_int, index=ori_data_pd.index, dtype='int')
    result.to_csv('./output/core_tensor/{}/core_tensor_{}_{}.csv'.format(station, tucker_rank, station),
                  header=None)