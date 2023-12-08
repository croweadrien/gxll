"""
作者：LIUYANGSHUO
日期：2022年10月12日
"""
import numpy as np
import pandas as pd
from pyod.models.knn import KNN  # imprt kNN分类器
from pyod.models.lof import LOF
from pyod.models.copod import COPOD
from pyod.models.iforest import IForest
from pyod.models.pca import PCA
from pyod.models.lscp import LSCP
from pyod.models.vae import VAE
from pyod.models.so_gaal import SO_GAAL
from pyod.models.ocsvm import OCSVM
from pyod.models.ecod import ECOD
from pyod.models.deep_svdd import DeepSVDD
from sklearn.metrics import precision_score, recall_score, f1_score

if __name__ == '__main__':
    station_name_list = ['ALL', 'ZL']
    # station_name = 'ALL'
    clf_name_list = ['KNN', 'LOF', 'IForest', 'PCA',  'SO_GAAL','OCSVM', 'ECOD','COPOD']
    clf_name_list = [  'DeepSVDD']
    for station_name in station_name_list:

        data = pd.read_excel('./OUTPUT/{}_data.xlsx'.format(station_name), index_col=0, header=None)
        label = pd.read_excel('./OUTPUT/{}_label.xlsx'.format(station_name), index_col=0, header=None)
        model_score = pd.DataFrame([], columns=['part_name', '召回率', '准确率', 'F1'])

        for clf_name in clf_name_list:
            check_result_pd = pd.DataFrame([])
            for i in range(len(data)):
                ts = data.iloc[i].values
                label_ts = label.iloc[i].values
                part_name = data.index[i]
                clf = eval(clf_name)()
                clf.fit(ts.reshape(-1, 1))
                check_result = clf.labels_  # 异常检测结果 array
                y_train_scores = clf.decision_scores_
                check_result_pd = pd.concat([check_result_pd, pd.DataFrame([check_result], index=[part_name])])
                r = recall_score(label_ts, check_result)
                p = precision_score(label_ts, check_result)
                f1score = f1_score(label_ts, check_result)
                score = pd.DataFrame([np.array([part_name, r, p, f1score])], index=[clf_name],
                                     columns=['part_name', '召回率', '准确率', 'F1'])
                model_score = pd.concat([model_score, score])

            check_result_pd.to_excel('./OUTPUT/对比实验/{}/{}_检测结果.xlsx'.format(station_name, clf_name))
        model_score.to_excel('./OUTPUT/对比实验/{}/模型得分.xlsx'.format(station_name))
