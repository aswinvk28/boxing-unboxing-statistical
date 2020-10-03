
import numpy as np
from copy import copy
import cvxpy as cvx
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import imageio
from sklearn.svm import SVR, LinearSVR
from sklearn.model_selection import GridSearchCV
import pandas as pd
import seaborn
import scipy
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

if __name__ == "__main__":

    np.random.seed(10)
    large_df = pd.DataFrame(columns=['problem_value', 'mean_r.value'])
    # f-statistic
    # fstat = np.random.uniform(1,100,(144,4))
    # cov = np.random.uniform(1,100,(10,4))
    inputs_ = np.random.normal(1,20,200)
    inputs = inputs_.copy()
    corr, corr2 = pd.DataFrame(np.random.uniform(1,100,(200,100))), pd.DataFrame(np.random.uniform(1,100,(288,100)))
    # chi-squared
    # corr, corr2 = None, None
    # corr, corr2 = pd.DataFrame(np.random.uniform(1,3,(144,144))), pd.DataFrame(np.random.uniform(1,3,(144,144)))
    # mahalanobis
    # corr, corr2 = pd.DataFrame(np.random.uniform(1,100,(144,144))), pd.DataFrame(np.random.uniform(1,100,(144,144)))
    # input_index = corr2.index.astype(np.int64).values
    # f-statistic
    yrj = np.matmul(corr2.T.corr('spearman').replace(np.nan, 0).values,corr2.values) # 144 x 4
    # chi-squared
    # yrj = corr.values * corr2.values # 144 x 144
    # mahalanobis
    # yrj = np.sqrt(np.square(np.matmul(corr.T.values,np.linalg.inv(np.cov(corr2.T.values)), corr.values))) # 144 x 144
    mean_r = cvx.Variable((corr2.values.shape[0],100))
    yro = corr2.values
    xio = inputs
    xij = np.random.normal(1,100,(200,100))
    vi = cvx.Variable((inputs.shape[0],1))
    objective = cvx.Minimize(cvx.sum(cvx.matmul(mean_r.T, yro)))
    constraints = [(cvx.sum(cvx.matmul(xij.T,vi).T)-cvx.sum(cvx.matmul(mean_r.T, yrj))) <= 0,
                   mean_r >= 0,cvx.matmul(vi.T, xio)==1]
    problem = cvx.Problem(objective, constraints)
    problem.solve(verbose=True, solver=cvx.SCS, max_iters=1000)
    df = pd.DataFrame(mean_r.value, columns=np.arange(mean_r.value.shape[1]))
    df['problem_value'] = [problem.value]*mean_r.value.shape[0]
    # df['corr2_index'] = input_index
    large_df = pd.concat([large_df,df], axis=0)
    large_df = large_df.reset_index()

    # seaborn.distplot(inputs_)

    ## f statistical value

    f_value = 0.0
    between_group = 4 * ((yro.mean(axis=0) - yro.mean())**2).sum() * 4 / 3
    within_group = xio.var(axis=0).sum() * 10/(10-1)
    f_value = between_group / within_group
    print("f_value: ", f_value)
