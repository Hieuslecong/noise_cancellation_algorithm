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
#######################################################################################################################################
#######################################################################################################################################

train_model_simulation = argparse.ArgumentParser(description='Image Classification Using PyTorch', usage='[option] model_name')
train_model_simulation.add_argument('--simulink_data', type=str, required=True)
train_model_simulation.add_argument('--path_model_save', type=str, required=True)
train_model_simulation.add_argument('--name_txt', type=str, required=True)
train_model_simulation.add_argument('--path_data_save', type=str, required=True)
train_model_simulation.add_argument('--path_data_mean', type=str, required=True)
args = train_model_simulation.parse_args()
#                 main

# load data
# load image crack
img = cv2.imread('D:/imgage_label/image_label.png')
#point_crack = np.argwhere(img == 255)
val_resize = 1
size_img = img.shape
new_width = int(size_img[1]*val_resize)
new_height = int(size_img[0]*val_resize)
img_resized = cv2.resize(src=img, dsize=(new_width, new_height),interpolation = cv2.INTER_AREA)
img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

img_resized[np.argwhere(img_resized>0)[:,0],np.argwhere(img_resized>0)[:,1]] = 1
skeleton = skeletonize(img_resized, method='lee') 
point_crack = np.argwhere(skeleton > 0)
# load data crack train
# data = pd.read_csv("F:/crack-machine/data_crack.txt",
#                    delimiter='\s+', index_col=False)
data = pd.read_csv(args.simulink_data,
                   index_col=False,sep=",")
data = (np.array(data))
cloumn1 = data[:, 0]
cloumn1 = np.resize(cloumn1, ([np.size(cloumn1), 1]))
cloumn2 = data[:, 1]
cloumn2 = np.resize(cloumn2, ([np.size(cloumn1), 1]))
point = np.argwhere(cloumn1 == 0)[:, 0]
point = np.append([-1], point)  # add the first crack loction

# dataframe test model
df_test = pd.DataFrame(
    {'test_x': point_crack[:, 0] / np.max(point_crack[:, 0]) 
     - (np.sum(point_crack[:, 0] / np.max(point_crack[:, 0])
               )) / len(point_crack[:, 0])+1,
     'test_y': point_crack[:, 1] / np.max(point_crack[:, 1])
     - (np.sum(point_crack[:, 1] / np.max(point_crack[:, 1]))) / len(point_crack[:, 0])+1})

# careate a list of crack names
number_crack = []
for i in range(0, len(point) - 1):
    number_crack_name = 'number_crack_%s' % i
    number_crack.append(number_crack_name)

# 3
# model shape

# LinearRegression degree 2
polynomial_features = PolynomialFeatures(degree=2,
                                         include_bias=False)
linear_regression = LinearRegression(n_jobs=-1)
pipeline_degree_2 = Pipeline([("polynomial_features", polynomial_features),
                              ("linear_regression", linear_regression)])

# model Ridge degree 1
ridge_degree_1 = make_pipeline(PolynomialFeatures(1), Ridge(alpha=1e-5))
# model Ridge degree 2
ridge_degree_2 = make_pipeline(PolynomialFeatures(2), Ridge(alpha=1e-5))
# model Ridge degree 3
ridge_degree_3 = make_pipeline(PolynomialFeatures(3), Ridge(alpha=1e-5))

# model ridge default
ridge_default = Ridge(alpha=0.0, random_state=0, normalize=True)

# model RidgeCV default
ridgeCV_default = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])

# model BayesianRidge
model_BayesianRidge = BayesianRidge(
    tol=1e-6, fit_intercept=False, compute_score=True)
init = [1, 1e-4]
model_BayesianRidge.set_params(alpha_init=init[0], lambda_init=init[1])


# model SGDRegressor
model_SGDRegressor = make_pipeline(StandardScaler(),
                                   SGDRegressor(max_iter=1000, tol=1e-3, random_state=True))

# model ARDRegression
model_ARDRegression = ARDRegression(compute_score=True)

# model MultiTask
model_MultiTask = MultiTaskElasticNet(alpha=0.1)

# model LassoLarsCV
model_LarsCV = LassoLarsCV(cv=5)

# Model KernelRidge
param_grid_KernelRidge = {"alpha": [1e0, 1e1, 1e-1, 1e-2]}
model_KernelRidge = GridSearchCV(
    KernelRidge(), param_grid=param_grid_KernelRidge)

# model gaussian karnel default
gp_kernel_1 = DotProduct() + WhiteKernel()
model_gaussian_kernel_default = GaussianProcessRegressor(kernel=gp_kernel_1)

# model graussian kernel
gp_kernel_2 = ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) \
    + WhiteKernel(1e-1)
model_gaussian_kernel_2 = GaussianProcessRegressor(kernel=gp_kernel_2)

# model gaussian correction
k1 = 66.0**2 * RBF(length_scale=67.0)  # long term smooth rising trend
k2 = 2.4**2 * RBF(length_scale=90.0) \
    * ExpSineSquared(length_scale=1.3, periodicity=1.0)  # seasonal component
# medium term irregularity
k3 = 0.66**2 \
    * RationalQuadratic(length_scale=1.2, alpha=0.78)
k4 = 0.18**2 * RBF(length_scale=0.134) \
    + WhiteKernel(noise_level=0.19**2)  # noise terms
kernel_gpml_1 = k1 + k2 + k3 + k4

model_gaussian_GPR = GaussianProcessRegressor(kernel=kernel_gpml_1, alpha=0,
                                              optimizer=None, normalize_y=True)

# Model gaussian correction 2
k1 = 50.0**2 * RBF(length_scale=50.0)  # long term smooth rising trend
k2 = 2.0**2 * RBF(length_scale=100.0) \
    * ExpSineSquared(length_scale=1.0, periodicity=1.0,
                     periodicity_bounds="fixed")  # seasonal component
# medium term irregularities
k3 = 0.5**2 * RationalQuadratic(length_scale=1.0, alpha=1.0)
k4 = 0.1**2 * RBF(length_scale=0.1) \
    + WhiteKernel(noise_level=0.1**2,
                  noise_level_bounds=(1e-5, np.inf))  # noise terms
kernel_2 = k1 + k2 + k3 + k4

model_gaussian_GPR_2 = GaussianProcessRegressor(kernel=kernel_2, alpha=0.5,
                                                normalize_y=True)

# model gaussian correction 3
kernel_3 = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
    + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
model_gaussian_RBF = GaussianProcessRegressor(kernel=kernel_3,
                                              alpha=0.0)

# model gaussion kernel
kernel_4 = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
model_gaussian_kernel_3 = GaussianProcessRegressor(
    kernel=kernel_4, n_restarts_optimizer=9)

# model gaussion kernel
kernel_4 = ConstantKernel(
    0.1, (0.01, 10.0)) * (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2)
model_gaussian_kernel_4 = GaussianProcessRegressor(kernel=kernel_4)

# model gaussion kernel
kernel_5 = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0),
                        nu=1.5)
model_gaussian_kernel_5 = GaussianProcessRegressor(kernel=kernel_5)

# model huber
huber = HuberRegressor(alpha=0.0, epsilon=1.35)

# model SVC
model_SVC = make_pipeline(StandardScaler(),
                          LinearSVR(random_state=0, tol=1e-6))

# model tree
model_tree = DecisionTreeRegressor(random_state=0)
# model_tree.fit(X, y, sample_weight=None, check_input=True,
#                  X_idx_sorted='deprecated')

# model SVR
model_SVR = GridSearchCV(SVR(kernel='rbf', gamma=0.1),
                         param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                                     "gamma": np.logspace(-2, 2, 5)})

# model KRR
model_KRR = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1),
                         param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                                     "gamma": np.logspace(-2, 2, 5)})

####################################################################################


# list model train
list_model_train = [('model_SVC', model_SVC),('Model_KernelRidge',model_KernelRidge),('Linear_SGD', model_SGDRegressor), 
('gaussian_kernel_2', model_gaussian_kernel_2),('gaussian_kernel_default', model_gaussian_kernel_default), ('Linear_deg_2', pipeline_degree_2),
                    ('Ridge_deg_1', ridge_degree_1), ('Ridge_deg_2', ridge_degree_2),
                    ('Ridge_deg_3', ridge_degree_3), ('Ridge_default', ridge_default),
                    ('RidgeCV_default', ridgeCV_default), ('Linear_bayes',
                                                           model_BayesianRidge),
                    
                    ('linear_ADR', model_ARDRegression), ('Model_MultiTask',
                                                          model_MultiTask),
                    ('Model_LarsCV', model_LarsCV),

                    ('gaussian_GPR', model_gaussian_GPR), ('gaussian_GPR_2',
                                                           model_gaussian_GPR_2),
                    ('gaussian_RBF', model_gaussian_RBF), ('gaussian_kernel_3',
                                                           model_gaussian_kernel_3),
                    ('gaussian_kernel_4', model_gaussian_kernel_4), ('gaussian_kernel_5',
                                                                     model_gaussian_kernel_5),
                    ('huber', huber), 
                    ('model_tree', model_tree), ('model_SVR',
                                                 model_SVR), ('model_KRR', model_KRR),
                    ]

########################################################################################################################################
########################################################################################################################################
#model 1
# list_MSE, list_MSA, list_RMSE, list_R_squared, list_Accuracy = [], [], [], [], []
# for i in range(0, len(point) - 1):
#     print(i)
#     data1 = data[point[i] + 1:point[i + 1], :]
#     #         # normalize data train
#     X = data1[:, 0] - np.sum(data1[:, 0]) / len(data1[:, 0]) + 1
#     X1 = np.resize(X, ([np.size(X), 1]))
#     y = data1[:, 1] - np.sum(data1[:, 1]) / len(data1[:, 0]) + 1
#     y1 = np.resize(y, ([np.size(y), 1]))
#     model1 = model_LarsCV
#     model1.fit(y1, X1)
#     MSA = mean_absolute_error(
#         X1, model1.predict(y1))
#     # MSE
#     MSE = mean_squared_error(
#         X1, model1.predict(y1))
#     # RMSE
#     RMSE = np.sqrt(MSE)
#     # R_squared
#     R_squared = r2_score(X1,
#                          model1.predict(y1))
#     list_MSE.insert(i, MSE)
#     list_MSA.insert(i, MSA)
#     list_RMSE.insert(i, RMSE)
#     list_R_squared.insert(i, R_squared)
# print(sum(list_RMSE)/len((list_RMSE)))
# data_out = {'MSE_' : list_MSE,
#             'RMSE_' : list_RMSE, 'MSA_' : list_MSA, 'R_squared_' : list_R_squared, }  # , 'Accuracy':list_Accuracy
# data_model = pd.DataFrame(data_out, index=number_crack)
# R = data_model['R_squared_'].values
# print((R[np.argwhere(R > 0.5)]).mean())


# print(hahhahaah)
# # export excel data model
# name_excel = 'F:/crack-machine/data_test.xlsx'
# data_model.to_excel(name_excel)
# i1=274
# data1 = data[point[i1] + 1:point[i1 + 1], :]
# #         # normalize data train
# X = data1[:, 0] - np.sum(data1[:, 0]) / len(data1[:, 0]) + 1
# X1 = np.resize(X, ([np.size(X), 1]))
# y = data1[:, 1] - np.sum(data1[:, 1]) / len(data1[:, 0]) + 1
# y1 = np.resize(y, ([np.size(y), 1]))
# model1 = model_LarsCV
# model1.fit(y1, X1)
# MSA = mean_absolute_error(
#     X1, model1.predict(y1))
# # MSE
# MSE = mean_squared_error(
#     X1, model1.predict(y1))
# # RMSE
# RMSE = np.sqrt(MSE)
# # R_squared
# R_squared = r2_score(X1,
#                      model1.predict(y1))
# # accuracy
# print(' crack number %s R_squared: %s MSA: %s MSE: %s' %
#       ( i1, R_squared, MSA, MSE))
# #model 2
# i2=401
# data1 = data[point[i2] + 1:point[i2 + 1], :]
# #         # normalize data train
# X = data1[:, 0] - np.sum(data1[:, 0]) / len(data1[:, 0]) + 1
# X2 = np.resize(X, ([np.size(X), 1]))
# y = data1[:, 1] - np.sum(data1[:, 1]) / len(data1[:, 0]) + 1
# y2 = np.resize(y, ([np.size(y), 1]))
# model2 = model_SVC
# model2.fit(y2, X2)

# # model 3
# i3=444
# data1 = data[point[i3] + 1:point[i3 + 1], :]
# #         # normalize data train
# X = data1[:, 0] - np.sum(data1[:, 0]) / len(data1[:, 0]) + 1
# X3 = np.resize(X, ([np.size(X), 1]))
# y = data1[:, 1] - np.sum(data1[:, 1]) / len(data1[:, 0]) + 1
# y3 = np.resize(y, ([np.size(y), 1]))
# model3 = huber
# model3.fit(y3, X3)

# # model 4

# i4=444
# data1 = data[point[i4] + 1:point[i4 + 1], :]
# #         # normalize data train
# X = data1[:, 0] - np.sum(data1[:, 0]) / len(data1[:, 0]) + 1
# X4 = np.resize(X, ([np.size(X), 1]))
# y = data1[:, 1] - np.sum(data1[:, 1]) / len(data1[:, 0]) + 1
# y4 = np.resize(y, ([np.size(y), 1]))
# model4 = ridge_default
# model4.fit(y4, X4)

# #model 5
# i5=274
# data1 = data[point[i5] + 1:point[i5 + 1], :]
# #         # normalize data train
# X = data1[:, 0] - np.sum(data1[:, 0]) / len(data1[:, 0]) + 1
# X5 = np.resize(X, ([np.size(X), 1]))
# y = data1[:, 1] - np.sum(data1[:, 1]) / len(data1[:, 0]) + 1
# y5 = np.resize(y, ([np.size(y), 1]))
# model5 = model_gaussian_kernel_default
# model5.fit(y5, X5)




# font_path = 'C:\Windows\Fonts\AGaramondPro-Regular.otf'
# font_prop = font_manager.FontProperties(fname=font_path, size=13)
# fig, ax = plt.subplots(figsize=(6, 6)) # Defines ax variable by creating an empty plot

# plt.plot(model1.predict(df_test[['test_x']]), -df_test[['test_x']],  'g-',alpha = 1, label= 'Model LarsCV', linewidth=1 )
# plt.plot(model2.predict(df_test[['test_x']]), -df_test[['test_x']],  'r-',alpha = 1, label= 'Model SVC', linewidth= 1 )
# plt.plot(model3.predict(df_test[['test_x']]), -df_test[['test_x']],  'k-',alpha = 1, label= 'Model huber', linewidth=1 )
# plt.plot(model4.predict(df_test[['test_x']]), -df_test[['test_x']],  'c-',alpha = 1, label= 'Model ridge_default', linewidth=1 )
# plt.plot(model5.predict(df_test[['test_x']]), -df_test[['test_x']],  'b-',alpha = 1, label= 'Model Gaussian \n kernel default', linewidth=1 )
# plt.scatter(df_test[['test_y']], -df_test[['test_x']],
#                     edgecolor='c', s=1, label="True crack")
# plt.scatter(X1, -y1, edgecolor='b', s=20, label="Crack samples")
# # Define the data to be plotted
# #plt.plot(x, y, 'b+', label='Data points')
# # plt.scatter(x2+1,y2,
# #         edgecolor='c', s=1, label="True crack")
# # plt.scatter(X,y, edgecolor='b', s=10, label="care 1 Crack samples  %s" %i)
# # plt.scatter(model.predict(y1),y2,edgecolor='k', s=1, label="Model train care1")
# plt.title("Top 5 Training model " , fontproperties=font_prop,
#                       size=13, verticalalignment='bottom')  # Size here overrides font_prop
# plt.xlabel("X", fontproperties=font_prop)
# plt.ylabel("Y", fontproperties=font_prop)
# #plt.text(0, 0, "Misc text", fontproperties=font_prop)
# # ax.legend(loc='best', prop={'size':'large'})
# lgd = plt.legend(loc="best", prop=font_prop)
# plt.axis('equal')
# fig.savefig('Top_5_best_trainning_model.png', dpi=125)
# plt.show()


# train model
k = 0
out_data_model = []
out_data_model_mean = []
list_name_model =[]
os.makedirs(args.path_data_mean + '/data_loss_mean_%s' %args.name_txt) 
os.makedirs(args.path_data_save + '/data_loss_%s' % (args.name_txt)) 

for name_model, model in list_model_train:
    list_name_model.append(name_model)
data_out_mean = {' model ': ['mean']} 
out_data_model_mean = pd.DataFrame(data_out_mean)
for name_model, model in list_model_train:
    print(name_model)
    print(args.path_model_save)
    folder_path = args.path_model_save + '/%s/%s/'%(args.name_txt,name_model)
    print(folder_path)
    #folder_path = 'F:/crack-machine/model new/cracktip8/%s' %name_model
    os.makedirs(folder_path)
    # train in data crack
    list_MSE, list_MSA, list_RMSE, list_R_squared, list_Accuracy = [], [], [], [], []
    list_MSE_mean, list_MAE_mean, list_RMSE_mean, list_R_squared_mean, list_Accuracy = [], [], [], [], [] 
    i = 0
    for i in range(0, len(point) - 1):
        data1 = data[point[i] + 1:point[i + 1], :]
        # normalize data train
        X = data1[:, 0] - (data1[:, 0]).mean() + 1
        X = np.resize(X, ([np.size(X), 1]))
        y = data1[:, 1] - (data1[:, 1]).mean() + 1
        y = np.resize(y, ([np.size(y), 1]))
        # train model in list
        model.fit(y, X)
        
        # if i<10:
        #     filename_model5 = 'F:/crack-machine/model new/cracktip8/%s/sample_crack_000%d.model' %(name_model,i)
        # elif i>=10 and i<100:
        #     filename_model5 = 'F:/crack-machine/model new/cracktip8/%s/sample_crack_00%d.model' %(name_model,i)
        # elif i>=100 and i<1000:
        #     filename_model5 = 'F:/crack-machine/model new/cracktip8/%s/sample_crack_0%d.model' %(name_model,i)
        # else:
        #     filename_model5 = 'F:/crack-machine/model new/cracktip8/%s/sample_crack_%d.model' %(name_model,i)
        if i<10:
            filename_model5 = args.path_model_save + '/%s/%s/sample_crack_000%d.model' %(args.name_txt,name_model,i)
        elif i>=10 and i<100:
            filename_model5 = args.path_model_save + '/%s/%s/sample_crack_00%d.model' %(args.name_txt,name_model,i)
        elif i>=100 and i<1000:
            filename_model5 = args.path_model_save + '/%s/%s/sample_crack_0%d.model' %(args.name_txt,name_model,i)
        else:
            filename_model5 = args.path_model_save + '/%s/%s/sample_crack_%d.model' %(args.name_txt,name_model,i)
        # save model 
        joblib.dump(model, filename_model5)
        # calculate ranting parameter
        # MSA
        MSA = mean_absolute_error(
            df_test[['test_y']], model.predict(df_test[['test_x']]))
        # MSE
        MSE = mean_squared_error(
            df_test[['test_y']], model.predict(df_test[['test_x']]))
        # RMSE
        RMSE = np.sqrt(MSE)
        # R_squared
        R_squared = r2_score(df_test[['test_y']],
                             model.predict(df_test[['test_x']]))
    #     # accuracy
    #     print('%s crack number %s R_squared: %s MSA: %s MSE: %s' %
    #           (name_model, i, R_squared, MSA, MSE))
    #     #print(R_squared )


    #     # plot model train 
    #     font_path = 'C:\Windows\Fonts\AGaramondPro-Regular.otf'
    #     font_prop = font_manager.FontProperties(fname=font_path, size=13) 
    #     fig, ax = plt.subplots(figsize=(6, 6)) # Defines ax variable by creating an empty plot

    #     # Define the data to be plotted
    #     #plt.plot(x, y, 'b+', label='Data points')
    #     plt.scatter(df_test[['test_y']], -df_test[['test_x']],
    #                 edgecolor='c', s=1, label="True crack")
    #     plt.scatter(X, -y, edgecolor='b', s=10, label="Crack samples %s" %i)
    #     plt.scatter(model.predict(df_test[['test_x']]),- df_test[[
    #                 'test_x']], edgecolor='r', s=0.51, label="Model train")


    #     # for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    #     #     label.set_fontproperties(font_prop)
    #     #     label.set_fontsize(13) # Size here overrides font_prop
    #     plt.title("Training model %s line" %name_model, fontproperties=font_prop,
    #               size=13, verticalalignment='bottom')  # Size here overrides font_prop
    #     plt.xlabel("X", fontproperties=font_prop)
    #     plt.ylabel("Y", fontproperties=font_prop)
    #     #plt.text(0, 0, "Misc text", fontproperties=font_prop)
         
    #     lgd = plt.legend(loc="best", prop=font_prop) # NB different 'prop' argument for legend
    #     #lgd.set_title("Legend", prop=font_prop)
    #     # plt.scatter(df_test[['test_y']], -df_test[['test_x']],
    #     #             edgecolor='c', s=1, label="model")
    #     # #plt.scatter((point_crack[:, 0] )/np.max(point_crack[:, 0]), (point_crack[:, 1] )/np.max(point_crack[:, 1] ),s=2, label="True 1 function")
    #     # plt.scatter(X, -y, edgecolor='b', s=1, label="Samples")
    #     # plt.scatter(model.predict(df_test[['test_x']]),- df_test[[
    #     #             'test_x']], edgecolor='r', s=1, label="test")
    #     # plt.xlabel("x")
    #     # plt.ylabel("y")
    #     # #plt.xlim((-0.5, 1.5))
    #     # #plt.ylim((-0.5, 1.5))
    #     #plt.legend(loc="best")
    #     #fig.savefig('filename.png', dpi=125)
    #     # plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
    #     #    degrees[0], -scores.mean(), scores.std()))

    #     plt.axis('equal')
    #     #fig.savefig('filename.png', dpi=125)
    #     plt.show()
    #     # add parameter to list
        list_MSE.insert(i, MSE)
        list_MSA.insert(i, MSA)
        list_RMSE.insert(i, RMSE)
        list_R_squared.insert(i, R_squared)
    
    data_out = {'MSE_' + name_model: list_MSE,
                    'RMSE_' + name_model: list_RMSE, 'MAE_' + name_model: list_MSA, 'R_squared_' + name_model: list_R_squared, }  # , 'Accuracy':list_Accuracy
    data_model = pd.DataFrame(data_out, index=number_crack)
    R = data_model['R_squared_' + name_model].values
    list_MSE_mean.append(sum(list_MSE)/len(list_MSE))
    list_MAE_mean.append(sum(list_MSA)/len(list_MSA))
    list_RMSE_mean.append(sum(list_RMSE)/len(list_RMSE))
    list_R_squared_mean.append((R[np.argwhere(R > 0.5)]).mean())
    data_out2 = {'MSE_' + name_model: list_MSE_mean,
                    'RMSE_' + name_model: list_RMSE_mean, 'MAE_' + name_model: list_MAE_mean, 'R_squared_' + name_model: list_R_squared_mean, }
    out_data_model_mean.loc[ : ,'MSE_1_' + name_model]       = list_MSE_mean
    out_data_model_mean.loc[ : ,'RMSE_1_' + name_model]      = list_RMSE_mean
    out_data_model_mean.loc[ : ,'MSA_1_' + name_model]       = list_MAE_mean
    out_data_model_mean.loc[ : ,'R_squared_1_' + name_model] = list_R_squared_mean
    # # export excel data model
    
    name_excel =  args.path_data_save + '/data_loss_%s/data_%s.xlsx' % (args.name_txt, name_model)
    data_model.to_excel(name_excel)

name_excel1 =  args.path_data_mean + '/data_loss_mean_%s/data_mean.xlsx' %args.name_txt
out_data_model_mean.to_excel(name_excel1)
########################################################################################################################################
########################################################################################################################################
