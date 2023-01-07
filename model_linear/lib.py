import numpy as np
import pandas as pd
import time
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.ticker
import joblib
import pickle
import math
from sklearn.cluster import DBSCAN
import random
from skimage import measure
from skimage.measure import label
import argparse
############################################################################################################
from sklearn import model_selection
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, ARDRegression, BayesianRidge, TheilSenRegressor, PassiveAggressiveRegressor, Ridge, RidgeCV, SGDRegressor, LogisticRegression, MultiTaskElasticNet
from sklearn.linear_model import LassoLarsCV, HuberRegressor, TheilSenRegressor, RANSACRegressor
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, DotProduct, RBF, ConstantKernel, Matern
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
from sklearn import neighbors
from skimage.morphology import skeletonize
import matplotlib.animation as animation
from matplotlib import style