import cv2
import matplotlib.pyplot as plt
import numpy as np
import joblib
# import pickle
import argparse
import pandas as pd
import os
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
from skimage.morphology import skeletonize
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
import random
from skimage import measure
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square
from skimage.measure import label
import matplotlib.font_manager as font_manager
# import multiprocessing
# import concurrent.futures
import time
import sys
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
############################################# function #####################################################

def return_data_crack(point_crack_turn, point_crack_stard):
    point_crack1 = point_crack_stard
    point_crack2 = point_crack_turn
    x1 = point_crack2[:, 0]
    x1 = np.resize(x1, ([np.size(x1), 1]))
    y1 = point_crack2[:, 1]
    y1 = np.resize(y1, ([np.size(y1), 1]))

    x = point_crack1[:, 0]
    x = np.resize(x, ([np.size(x), 1]))
    y = point_crack1[:, 1]
    y = np.resize(y, ([np.size(y), 1]))
    AB = x[0] - x[-1]
    BC = y[0] - y[-1]

    AB1 = x1[0] - x1[-1]
    BC1 = y1[0] - y1[-1]
    if BC1 == 0:
        BC1 = 0.000000000001
    if BC == 0:
        BC = 0.000000000001
    if AB1 == 0:
        AB1 = 0.000000000001
    if AB == 0:
        AB = 0.000000000001
    alpha = math.atan(BC / AB)
    alpha1 = math.atan(BC1 / AB1)
    if alpha < 0:
        alpha = alpha + math.pi
    if alpha1 < 0:
        alpha1 = alpha1 + math.pi
    if alpha > alpha1:
        delta1 = alpha - alpha1
        delta2 = math.pi + delta1
    if alpha1 > alpha:
        delta1 = math.pi - abs(alpha - alpha1)
        delta2 = math.pi + abs(delta1)
    tam_x = ((np.max(x) - np.min(x)) / 2 + np.min(x))
    tam_y = ((np.max(y) - np.min(y)) / 2 + np.min(y))
    #tam_x = x1[round(len(x1)/2)]

    x = x - ((np.max(x) - np.min(x)) / 2 + np.min(x))
    y = y - ((np.max(y) - np.min(y)) / 2 + np.min(y))
    x1 = x1 - ((np.max(x1) - np.min(x1)) / 2 + np.min(x1))
    y1 = y1 - ((np.max(y1) - np.min(y1)) / 2 + np.min(y1))
    #plt.scatter(x, y)
    x_end = x1 * math.cos(delta1) - y1 * math.sin(delta1) + tam_x
    y_end = x1 * math.sin(delta1) + y1 * math.cos(delta1) + tam_y

    # TH2
    #plt.scatter(x1, y1)
    x_end2 = x1 * math.cos(delta2) - y1 * math.sin(delta2) + tam_x
    y_end2 = x1 * math.sin(delta2) + y1 * math.cos(delta2) + tam_y
    MAE = mean_absolute_error(x, x_end)
    MAE2 = mean_absolute_error(x, x_end2)
    # MSE
    MSE = mean_squared_error(x, x_end)
    MSE2 = mean_squared_error(x, x_end2)
    # RMSE
    RMSE = np.sqrt(mean_squared_error(x, x_end))
    RMSE2 = np.sqrt(mean_squared_error(x, x_end2))
    # R squared 
    R2 = r2_score(x, x_end)
    R22 = r2_score(x, x_end2)
    if R2 <-1:
        R2 = 0.00001
    if R22 <-1:
        R22 = 0.00001
    print('R1 :%s R2: %s'%(R2,R22))
    M1 = MAE + RMSE + abs(1/R2)
    M2 = MAE2 + RMSE2 + abs(1/R22)
    print('M1 :%s M2: %s'%(M1,M2))
    if M1 < M2:
        return x_end, y_end
    else:
        return x_end2, y_end2
def clear():
    os.system(" cls ")


def check_crack(image_input):
    return len(measure.regionprops(label(image_input, connectivity=2)))


def single_crack(point_crack):
    clustering = DBSCAN(eps=5, min_samples=12).fit(point_crack)
    return np.unique(clustering.labels_)


def unify_data_crack(point_crack_stard, point_crack_turn):
    point_crack1 = point_crack_stard
    point_crack2 = point_crack_turn
    x = point_crack2[:, 0] - (point_crack2[:, 0]).mean() + 1
    x = np.resize(x, ([np.size(x), 1]))
    y = point_crack2[:, 1] - (point_crack2[:, 1]).mean() + 1
    y = np.resize(y, ([np.size(y), 1]))

    x1 = point_crack1[:, 0]
    x1 = np.resize(x1, ([np.size(x1), 1]))
    y1 = point_crack1[:, 1]
    y1 = np.resize(y1, ([np.size(y1), 1]))

    AB = x[0] - x[-1]
    BC = y[0] - y[-1]

    AB1 = x1[0] - x1[-1]
    BC1 = y1[0] - y1[-1]
    if BC1 == 0:
        BC1 = 0.000000000001
    if BC == 0:
        BC = 0.000000000001
    if AB1 == 0:
        AB1 = 0.000000000001
    if AB == 0:
        AB = 0.000000000001
    alpha = math.atan(BC / AB)
    alpha1 = math.atan(BC1 / AB1)
    if alpha < 0:
        alpha = alpha + math.pi
    if alpha1 < 0:
        alpha1 = alpha1 + math.pi
    if alpha > alpha1:
        delta1 = alpha - alpha1
        delta2 = math.pi + delta1
    if alpha1 > alpha:
        delta1 = math.pi - abs(alpha - alpha1)
        delta2 = math.pi + abs(delta1)
    x1 = x1 - x1[round(len(x1) / 2)]
    y1 = y1 - y1[round(len(y1) / 2)]
    x_end = x1 * math.cos(delta1) - y1 * \
        math.sin(delta1) + x[round(len(x) / 2)]
    y_end = x1 * math.sin(delta1) + y1 * \
        math.cos(delta1) + y[round(len(y) / 2)]
    x_end2 = x1 * math.cos(delta2) - y1 * \
        math.sin(delta2) + x[round(len(x) / 2)]
    y_end2 = x1 * math.sin(delta2) + y1 * \
        math.cos(delta2) + y[round(len(y) / 2)]

    # df_crack_trun = pd.DataFrame({'X_trun_1':x_end[:,np.newaxis],'y_trun_1':y_end[:,np.newaxis],'X_trun_2':x_end2[:,np.newaxis],'y_trun_2':y_end2[:,np.newaxis]})
    return x_end, y_end, x_end2, y_end2


def check_crack(image_input):
    return len(measure.regionprops(label(image_input, connectivity=2)))


def single_crack(point_crack):
    clustering = DBSCAN(eps=5, min_samples=12).fit(point_crack)
    return np.unique(clustering.labels_)


def best_model_crack(point_crack):
    if len(single_crack(point_crack)) >= 2:
        print('No single crack')
        return
    if np.size(point_crack) < 1:
        print('image emtpy')
        return
    x_son = point_crack[:, 0] / np.max(point_crack[:, 0]) - (
        point_crack[:, 0] / np.max(point_crack[:, 0])).mean() + 1
    x_son = np.resize(x_son, ([np.size(x_son), 1]))
    y_son = point_crack[:, 1] / np.max(point_crack[:, 1]) - (
        point_crack[:, 1] / np.max(point_crack[:, 1])).mean() + 1
    y_son = np.resize(y_son, ([np.size(y_son), 1]))
    point_crack = np.hstack([x_son, y_son])
    MIN_RMSE = 10
    global data, path_model_folder_input
    crack_predict = []
    cloumn1 = data[:, 0]
    cloumn1 = np.resize(cloumn1, ([np.size(cloumn1), 1]))
    cloumn2 = data[:, 1]
    cloumn2 = np.resize(cloumn2, ([np.size(cloumn1), 1]))
    point = np.argwhere(cloumn1 == 0)[:, 0]
    point = np.append([-1], point)  # add the first crack loction
    folder_path_1 =path_model_folder_input
    for path_folder_model in os.listdir(folder_path_1):
        print(path_folder_model)
        name_model = path_folder_model
        path_folder_model = path_model_folder_input +'/'+ name_model
        print(path_folder_model)
        # train model
        k = 0
        out_data_max_model = []
        for path_model in os.listdir(path_folder_model):
            # print(path_model)
            path_model = path_folder_model + "/" + path_model
            # print(path_model)
            model = joblib.load(path_model)
            # train in data crack
            i = int(path_model[-9:-6])
            # print(int(path_model[-9:-6]))
            data1 = data[point[i] + 1:point[i + 1], :]
            x1, y1, x2, y2 = unify_data_crack(point_crack, data1)
            X = data1[:, 0] - (data1[:, 0]).mean() + 1
            X = np.resize(X, ([np.size(X), 1]))
            y = data1[:, 1] - (data1[:, 1]).mean() + 1
            y = np.resize(y, ([np.size(y), 1]))
            RMSE_1 = np.sqrt(mean_squared_error(x1, model.predict(y1)))
            RMSE_2 = np.sqrt(mean_squared_error(x2, model.predict(y2)))
            R21 = R_squared = r2_score(x1, model.predict(y1))
            R22 = R_squared = r2_score(x2, model.predict(y2))
            if RMSE_1 < MIN_RMSE:
                MIN_RMSE = RMSE_1
                x2_1 = np.reshape(model.predict(
                    y1), (len(model.predict(y1)), 1))
                crack_predict = np.hstack([x2_1, y1])
                care = R21

            if RMSE_2 < MIN_RMSE:
                MIN_RMSE = RMSE_2
                x2_1 = np.reshape(model.predict(
                    y2), (len(model.predict(y2)), 1))
                crack_predict = np.hstack([x2_1, y2])
                care = R22
    return MIN_RMSE, crack_predict, care


def list_R2_model_each_secment(name_model, point_crack):
    R2_model = {}
    global data, path_model_folder_input
    cloumn1 = data[:, 0]
    cloumn1 = np.resize(cloumn1, ([np.size(cloumn1), 1]))
    point = np.argwhere(cloumn1 == 0)[:, 0]
    point = np.append([-1], point)
    path_folder_model = path_model_folder_input + name_model
    a = int(np.round(len(os.listdir(path_folder_model)) * 0.35))
    # print(a)
    list_model_test = random.sample(os.listdir(path_folder_model), k=a)
    list_test = 0
    for path_model in (list_model_test):
        # print(path_model)
        path_model = path_folder_model + "/" + path_model
        # print(path_model)
        model = joblib.load(path_model)
        # train in data crack
        i = int(path_model[-9:-6])
        # print(int(path_model[-9:-6]))
        data1 = data[point[i] + 1:point[i + 1], :]
        x1, y1, x2, y2 = unify_data_crack(point_crack, data1)
        R_squared_1 = np.sqrt(mean_squared_error(x1, model.predict(y1)))
        R_squared_2 = np.sqrt(mean_squared_error(x2, model.predict(y2)))

        list_test = np.hstack((list_test, R_squared_1))
        list_test = np.hstack((list_test, R_squared_2))
        # if R_squared_1 > max_R2:
        #     max_R2 = R_squared_1
        #     model_best = path_model
        #     data_crak_train = data1
        # if R_squared_2 > max_R2:
        #     max_R2 = R_squared_2
        #     model_best = path_model
        #     data_crak_train = data1
    R2_model[name_model] = sum(list_test) / len(list_test)
    list_test = 0
    return R2_model

def nomalize_data_out(data_input):
    scaler = MinMaxScaler() 
    data_input[['RMSE_nor','MAE_nor']] = scaler.fit_transform(data_input[['RMSE','MAE']]) 
    data_input[['1/R_squared']] = (1-data_input[['R_squared']]) 
    data_input[['R_squared_nor']] = scaler.fit_transform(abs(1-data_input[['R_squared']]))
    data_input['sum'] =  (abs(data_input['R_squared_nor']) + data_input['RMSE_nor'] + data_input['MAE_nor'])
    RMSE_max = data_input['RMSE'][data_input['sum'].idxmin()]
    MAE_max = data_input['MAE'][data_input['sum'].idxmin()]
    R2_max = data_input['R_squared'][data_input['sum'].idxmin()]
    num_max = data_input['num'][data_input['sum'].idxmin()]
    return RMSE_max,MAE_max,R2_max,num_max,data_input
def nomalize_data_out2(data_input):
    scaler = MinMaxScaler() 
    data_input[['RMSE_nor','MAE_nor']] = scaler.fit_transform(data_input[['RMSE','MAE']]) 
    data_input[['1/R_squared']] = (1-data_input[['R_squared']]) 
    data_input[['R_squared_nor']] = scaler.fit_transform(abs(1-data_input[['R_squared']]))
    data_input['sum'] =  (abs(data_input['R_squared_nor']) + data_input['RMSE_nor'] + data_input['MAE_nor'])
    return data_input

def return_predict_crack_best(point_crack,data_input):
    global data, path_model_folder_input
    cloumn1 = data[:, 0]
    cloumn1 = np.resize(cloumn1, ([np.size(cloumn1), 1]))
    cloumn2 = data[:, 1]
    cloumn2 = np.resize(cloumn2, ([np.size(cloumn1), 1]))
    point = np.argwhere(cloumn1 == 0)[:, 0]
    point = np.append([-1], point)  
    #
    path_folder_model = path_model_folder_input
    #
    #index_floud = (data_input['sum'].idxmin() // num_model) -1
    name_model = data_input['model'][data_input['sum'].idxmin()]
    num_model = data_input['num'][data_input['sum'].idxmin()]
    #
    path_folder_model = path_model_folder_input + name_model
    #
    index_model = data_input['num'][data_input['sum'].idxmin()]
    path_model_best = os.listdir(path_folder_model)[index_model]
    #
    model = joblib.load(path_folder_model + "/" +path_model_best)
    i = int((path_folder_model + "/" +path_model_best)[-9:-6])
    
    #
    data1 = data[point[i] + 1:point[i + 1], :]
    x1, y1, x2, y2 = unify_data_crack(point_crack, data1)
    X = data1[:, 0] - (data1[:, 0]).mean() + 1
    X = np.resize(X, ([np.size(X), 1]))
    y = data1[:, 1] - (data1[:, 1]).mean() + 1
    y = np.resize(y, ([np.size(y), 1]))
    # MAE 
    MAE = mean_absolute_error(x1, model.predict(y1))
    MAE2 = mean_absolute_error(x2, model.predict(y2))
    # RMSE
    RMSE = np.sqrt(mean_squared_error(x1, model.predict(y1)))
    RMSE2 = np.sqrt(mean_squared_error(x2, model.predict(y2)))
    # R squared 
    R2 = r2_score(x1, model.predict(y1))
    R22 = r2_score(x2, model.predict(y2))
    if R2 <-1:
        R2 = 0.00001
    if R22 <-1:
        R22 = 0.00001
    print('R1 :%s R2: %s'%(R2,R22))
    M1 = MAE + RMSE + abs(1/R2)
    M2 = MAE2 + RMSE2 + abs(1/R22)
    print('M1 :%s M2: %s'%(M1,M2))
    if M1 < M2:
        return RMSE, MAE, R2
    else:
        return RMSE2, MAE2, R22
    

    
def best_model_crack(point_crack):
    global data,path_model_folder_input, max_R2, crack_predict, care,data
    max_R2 = -1
    
    cloumn1 = data[:, 0]
    cloumn1 = np.resize(cloumn1, ([np.size(cloumn1), 1]))
    cloumn2 = data[:, 1]
    cloumn2 = np.resize(cloumn2, ([np.size(cloumn1), 1]))
    point = np.argwhere(cloumn1 == 0)[:, 0]
    point = np.append([-1], point)  # add the first crack loction

    folder_path_1 = path_model_folder_input
    max_R22 = 0
    i=0
    list_num_max, list_model, list_MAE_max, list_RMSE_max, list_R_squared_max, list_R2_max, length_crack = [],[], [], [], [], [],[]
    for path_folder_model in os.listdir(folder_path_1):
        print(path_folder_model)
        name_model = path_folder_model
        path_folder_model = path_model_folder_input + '/'+ name_model
        # print(path_folder_model)
        #number_crack.append(name_model)
        # train model
        k = 0
        out_data_max_model = []
        number_crack = []
        
        list_num, list_MAE, list_RMSE, list_R_squared, list_R2 = [], [], [], [], []
        
        for path_model in os.listdir(path_folder_model):
            
            
            # print(path_model)
            path_model = path_folder_model + "/" + path_model
            # print(path_model)
            model = joblib.load(path_model)
            # train in data crack
            i = int(path_model[-9:-6])
            # print(int(path_model[-9:-6]))

            data1 = data[point[i] + 1:point[i + 1], :]
            x1, y1, x2, y2 = unify_data_crack(point_crack, data1)
            X = data1[:, 0] - (data1[:, 0]).mean() + 1
            X = np.resize(X, ([np.size(X), 1]))
            y = data1[:, 1] - (data1[:, 1]).mean() + 1
            y = np.resize(y, ([np.size(y), 1]))
            R_squared_1 = r2_score(x1, model.predict(y1))
            R_squared_2 = r2_score(x2, model.predict(y2))

            
            MAE_1 = mean_absolute_error(x1, model.predict(y1))
            # MSE
            MSE_1 = mean_squared_error(x1, model.predict(y1))
            # RMSE
            RMSE_1 = np.sqrt(MSE_1)
            MAE_2 = mean_absolute_error(x2, model.predict(y2))
            # MSE
            MSE_2 = mean_squared_error(x2, model.predict(y2))
            # RMSE
            RMSE_2 = np.sqrt(MSE_2)
            
            if R_squared_1> 0 or R_squared_2>0:
                number_crack.append(name_model)
                list_num.append(i)
                if R_squared_1>=R_squared_2:
                    list_MAE.append( MAE_1)
                    list_RMSE.append(RMSE_1)
                    list_R2.append(R_squared_1)

                    
                else  :
                    list_MAE.append( MAE_2)
                    list_RMSE.append(RMSE_2)
                    list_R2.append(R_squared_2)

                if R_squared_1 > max_R22:
                    max_R22 = R_squared_1
                    x2_1 = np.reshape(model.predict(
                        y1), (len(model.predict(y1)), 1))
                    crack_predict2 = np.hstack([x2_1, y1])
                    care = 1

                if R_squared_2 > max_R22:
                    max_R22 = R_squared_2
                    x2_1 = np.reshape(model.predict(
                        y2), (len(model.predict(y2)), 1))
                    crack_predict2 = np.hstack([x2_1, y2])
                    care = 2
        if len(list_MAE)>0:
            data_out_1 = {'num':list_num,'model':number_crack,'RMSE' : list_RMSE, 'MAE' : list_MAE, 'R_squared': list_R2, }  # , 'Accuracy':list_Accuracy
            data_model_1 = pd.DataFrame(data_out_1)
            max_RMSE, max_MAE, max_R2,max_num,data_model_1 = nomalize_data_out(data_model_1)
            list_MAE_max.append( max_MAE)
            list_RMSE_max.append(max_RMSE)
            list_R2_max.append(max_R2)
            list_num_max.append(max_num)
            list_model.append(name_model)
        else: continue
    if len(list_num_max)>0:
        print(max_R22)
        data_out_2 = {'num':list_num_max,'model':list_model,'RMSE' : list_RMSE_max, 'MAE' : list_MAE_max, 'R_squared': list_R2_max, }  # , 'Accuracy':list_Accuracy
        data_model_2 = pd.DataFrame(data_out_2)
        #print(data_model_1.max(axis=0))
        max_RMSE, max_MAE, max_R2,max_num,data_model_2 = nomalize_data_out(data_model_2)
        #RMSE, MAE, R2 = return_predict_crack_best(point_crack,data_model_2)
        print(max_RMSE, max_MAE, max_R2,max_num)
        
        # name_excel_data_mean2 = 'E:/data_image_label3.xlsx' 
        # data_model_1.to_excel(name_excel_data_mean2)
        return  max_RMSE, max_MAE, max_R2
    else: return  [], [], []

##########################################################################################################
##########################################################################################################
############################################################### main #######################################

test_model_simulation =argparse.ArgumentParser(description='test model crack ', usage='[option] model_name')
test_model_simulation.add_argument('--path_gt', type=str,default='', required=True)
test_model_simulation.add_argument('--path_model', type=str,default='', required=True)
test_model_simulation.add_argument('--start_img', type=int, default=0,required=True)
test_model_simulation.add_argument('--end_img', type=str, default='all',required=True)
test_model_simulation.add_argument('--path_data_txt', type=str,default='', required=True)
test_model_simulation.add_argument('--save_data', default=True, required=True)
test_model_simulation.add_argument('--path_data_mean', type=str,default='', required=True)
test_model_simulation.add_argument('--name_data_set', type=str,default='', required=True)
args = test_model_simulation.parse_args()

path_model_folder_input = args.path_model
list_path_image = []
photo_tail = []
data = pd.read_csv(args.path_data_txt,
                       delimiter=',', index_col=False)
data = (np.array(data))
###############################################################################################################3
# path folder data test
path_folder_data_test = args.path_gt
num = 0
list_path_image = []
photo_tail = []
for filename in os.listdir(path_folder_data_test):
    print(filename)
    num = num + 1
    print(' image %s / %s' % (num, len(os.listdir(path_folder_data_test))))
    if num == 1:
        photo_tail = filename[-4:len(filename)]
    name_image = filename[0:-4]
    if filename[-4:len(filename)] != photo_tail:
        continue
    img_path = path_folder_data_test + '/' + name_image + photo_tail
    list_path_image.append((filename, img_path))

list_MSE, list_MAE, list_RMSE, list_R_squared, list_Accuracy, list_name_image ,length_crack= [],[
], [], [], [], [], []
num = 0
if args.end_img == 'all':
    start = args.start_img
    end = len(list_path_image)
else:
    start = args.start_img
    end = int(args.end_img)
for filename, path_image in list_path_image[start:end]:
    print(filename)
    print(path_image)
    num = num + 1
    print(' image %s / %s' % (num, len(list_path_image[start:end])))
    name_image = filename[0:-4]


    # print(filename[-4:len(filename)])
    if filename[-4:len(filename)] != photo_tail:
        continue
    img_path = path_folder_data_test + '/' + name_image + photo_tail
    # print(img_path)
    img_input = cv2.imread(img_path)
    skeleton = skeletonize(img_input)
    skeleton = cv2.cvtColor(skeleton, cv2.COLOR_BGR2GRAY)
    # check one crack
    if check_crack(skeleton) >= 2:
        continue

    # load data
    # load image crack

    point_crack = np.argwhere(skeleton > 0)
    # if len(point_crack[:,0]) < 30 :
    #     continue
    # check singel crack
    # check emtpy crack image
    if np.size(point_crack) < 1:
        print('image emtpy')
        continue
    if len(single_crack(point_crack)) >= 2:
        continue
    if abs(len(np.unique(point_crack[:, 0])) - len(point_crack[:, 0])) > abs(len(np.unique(point_crack[:, 1])) - len(point_crack[:, 1])):
        columnIndex = 1
        # Sort 2D numpy array by 2nd Column
        point_crack = point_crack[point_crack[:,
                                              columnIndex].argsort()]
    else:
        columnIndex = 0
        # Sort 2D numpy array by 2nd Column
        point_crack = point_crack[point_crack[:,
                                              columnIndex].argsort()]
    # if len(point_crack[:,0]) < 30 :
    #     continue
    # check singel crack
    # check emtpy crack image
    x_son = point_crack[:, 0] / np.max(point_crack[:, 0]) - (
    point_crack[:, 0] / np.max(point_crack[:, 0])).mean() + 1
    x_son = np.resize(x_son, ([np.size(x_son), 1]))
    y_son = point_crack[:, 1] / np.max(point_crack[:, 1]) - (
        point_crack[:, 1] / np.max(point_crack[:, 1])).mean() + 1
    y_son = np.resize(y_son, ([np.size(y_son), 1]))
    point_crack = np.hstack([x_son, y_son])
    RMSE, MAE, R2 = best_model_crack(point_crack)
    if np.size(RMSE)==0: continue
    list_name_image.append(name_image)
if args.save_data == 'True':
    import datetime
    date_object = datetime.date.today()
    try:
        os.makedirs(args.path_data_mean +'_%s'%(date_object))
    except:
        print('done make folder') 
    path_save=args.path_data_mean + '/' +'_%s_'%(date_object)   
    list_MAE.append(MAE)
    list_RMSE.append(RMSE)
    list_R_squared.append(R2)
    length_crack.append(len(y_son))
    data_out_1 = {'Name_image': list_name_image,'RMSE': list_RMSE, 'MAE': list_MAE, 'R_squared': list_R_squared, }  # , 'Accuracy':list_Accuracy
    name_excel_data_mean = path_save + '/data_mean_%s_crack_%d_to_%d_.xlsx' %(args.name_data_set,start,end)
    print(name_excel_data_mean)
    #name_excel_data_mean = 'F:/crack-machine/data excel best model paragraph/data_mean_crack500_path5.xlsx'
    data_model_1 = pd.DataFrame(data_out_1)
    data_model_1.to_excel(name_excel_data_mean)
    # plt.scatter(point_crack[:,0], point_crack[:,1],edgecolor='r', s=2, label=" True crack ")
    # #print(list_max_model)
    # plt.legend(loc="best")
    # plt.show()