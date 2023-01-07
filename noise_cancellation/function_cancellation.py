from .lib import *
from model_linear.function import *
from add_noise.noise_crack import *
def clear():
    os.system(" cls ")

def check_crack(image_input):
    return len(measure.regionprops(label(image_input, connectivity=2)))

def single_crack(point_crack):
    clustering = DBSCAN(eps=5, min_samples=12).fit(point_crack)
    return np.unique(clustering.labels_)

def unify_data_crack_fc(point_crack_stard, point_crack_turn):
    point_crack1 = point_crack_stard
    point_crack2 = point_crack_turn
    x = point_crack2[:, 0] #- (point_crack2[:, 0]).mean() + 1
    x = np.resize(x, ([np.size(x), 1]))
    y = point_crack2[:, 1] #- (point_crack2[:, 1]).mean() + 1
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
    else:
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

def nomalize_data_out(data_input):
    scaler = MinMaxScaler() 
    #print(data_input)
    data_input[['RMSE_nor','MAE_nor']] = scaler.fit_transform(data_input[['RMSE','MAE']]) 
    data_input[['1/R_squared']] = (1-data_input[['R_squared']]) 
    data_input[['R_squared_nor']] = scaler.fit_transform(abs(1-data_input[['R_squared']]))
    data_input['sum'] =  (abs(data_input['R_squared_nor']) + data_input['RMSE_nor'] + data_input['MAE_nor'])
    RMSE_max = data_input['RMSE'][data_input['sum'].idxmin()]
    MAE_max = data_input['MAE'][data_input['sum'].idxmin()]
    R2_max = data_input['R_squared'][data_input['sum'].idxmin()]
    index_model = data_input['num'][data_input['sum'].idxmin()]
    name_model_best = data_input['name_model'][data_input['sum'].idxmin()]
    return RMSE_max,MAE_max,R2_max,index_model,data_input,name_model_best

def return_predict_crack_best(point_crack,name_model,index_model,data_simulation,path_folder_model,Snr_db):
    cloumn1 = data_simulation[:, 0]
    cloumn1 = np.resize(cloumn1, ([np.size(cloumn1), 1]))
    cloumn2 = data_simulation[:, 1]
    cloumn2 = np.resize(cloumn2, ([np.size(cloumn1), 1]))
    point = np.argwhere(cloumn1 == 0)[:, 0]
    point = np.append([-1], point)  
    
    #index_floud = (data_input['sum'].idxmin() // num_model) -1
    print(name_model)
    print(index_model)
    #
    path_model = path_folder_model+'/' + name_model
    #
    path_model_best = os.listdir(path_model)[index_model]
    print(path_folder_model + "/" +path_model_best)
    #
    model = joblib.load(path_model + "/" +path_model_best)
    i = int((path_folder_model + "/" +path_model_best)[-9:-6])
    data_simulation_son = data_simulation[point[i] + 1:point[i + 1], :]
    X = nomalize_crack_simulation(data_simulation_son[:, 0])
    y = nomalize_crack_simulation(data_simulation_son[:, 1] )
    # add noise
    x_simu,y_simu=add_noise(X,y,Snr_db=Snr_db,num_point=100)
    x_crack_1,y_crack_1,x_crack_2,y_crack_2,x_simu,y_simu=nomalize_data_crack_to_simulation(np.hstack((x_simu,y_simu)),point_crack)# R_squared
    # R_squared
    R2_1 = r2_score(y_crack_1, model.predict(x_crack_1))
    R2_2 = r2_score(y_crack_2, model.predict(x_crack_2))
    # RMSE
    RMSE_1 = np.sqrt(mean_squared_error(y_crack_1, model.predict(x_crack_1)))
    RMSE_2 = np.sqrt(mean_squared_error(y_crack_2, model.predict(x_crack_2)))
    # MAE
    MAE_1 = mean_absolute_error(y_crack_1, model.predict(x_crack_1))
    MAE_2 = mean_absolute_error(y_crack_2, model.predict(x_crack_2))
    
    M1 = MAE_1 + RMSE_1 + abs(1-R2_1)
    M2 = MAE_2 + RMSE_2 + abs(1-R2_2)
    #print('M1 :%s M2: %s'%(M1,M2))
    if M1 < M2:
        #print('RMSE :%s MAE : %s R2: %s'%(RMSE,MAE,R2))
        x2_1 = np.reshape(model.predict(x_crack_1), (len(model.predict(x_crack_1)), 1))
        crack_predict = np.hstack([x_crack_1,x2_1])
        crack_nor = np.hstack([x_crack_1, y_crack_1])
    else:
        #print('RMSE :%s MAE : %s R2: %s'%(RMSE2,MAE2,R22))
        x2_1 = np.reshape(model.predict(x_crack_2), (len(model.predict(x_crack_2)), 1))
        crack_predict = np.hstack([x_crack_2,x2_1])
        crack_nor = np.hstack([x_crack_2, y_crack_2])
    return crack_predict,crack_nor


def find_model_best_crack(data_simulation,data_crack,path_model,Snr_db,num_point_add):
    ###############
    
    point = np.argwhere(data_simulation[:,0] == 0)[:, 0]
    point = np.append([-1], point)
    list_num_max, list_model, list_MAE_max, list_RMSE_max, list_hist_model, list_R2_max = [], [], [], [], [],[]
    for name_model in os.listdir(path_model):
        #print(name_model)
        path_folder_model = path_model+'/' + name_model
        k = 0
        out_data_max_model, list_name_model = [], []
        list_num, list_MAE, list_RMSE, list_R_squared, list_R2 = [], [], [], [], []
        for path_model_son in os.listdir(path_folder_model):
            #print(path_model_son)
            
            path_model_son = path_folder_model + "/" + path_model_son
            print(path_model_son)
            model = joblib.load(path_model_son)
            i = int(path_model_son[-9:-6])
            data_simulation_son = data_simulation[point[i] + 1:point[i + 1], :]
            X = nomalize_crack_simulation(data_simulation_son[:, 0])
            y = nomalize_crack_simulation(data_simulation_son[:, 1] )
            # X[np.isnan(X)]=1
            # y[np.isnan(y)]=1
            # add noise
            x_simu,y_simu=add_noise(X,y,Snr_db=Snr_db,num_point=num_point_add)
            x_crack_1,y_crack_1,x_crack_2,y_crack_2,x_simu,y_simu=nomalize_data_crack_to_simulation(np.hstack((x_simu,y_simu)),data_crack)# R_squared
            # plt.plot(x_crack_1,y_crack_1)
            # plt.plot(x_crack_1, model.predict(x_crack_1))
            # plt.show()
            R2_1 = r2_score(y_crack_1, model.predict(x_crack_1))
            R2_2 = r2_score(y_crack_2, model.predict(x_crack_2))
            # RMSE
            RMSE_1 = np.sqrt(mean_squared_error(y_crack_1, model.predict(x_crack_1)))
            RMSE_2 = np.sqrt(mean_squared_error(y_crack_2, model.predict(x_crack_2)))
            # MAE
            MAE_1 = mean_absolute_error(y_crack_1, model.predict(x_crack_1))
            MAE_2 = mean_absolute_error(y_crack_2, model.predict(x_crack_2))
            
            M1 = MAE_1 + RMSE_1 + abs(1-R2_1)
            M2 = MAE_2 + RMSE_2 + abs(1-R2_2)
            
            list_name_model.append(name_model)
            list_num.append(i)

            if M1 < M2:
                print(('RMSE :%.5f MAE: %.5f 1-R2: %.5f'%(RMSE_1,MAE_1,1-R2_1)))
                list_MAE.append( MAE_1)
                list_RMSE.append(RMSE_1)
                list_R2.append(R2_1)
                if R2_1>0.8:
                    list_hist_model.append(name_model)
                #return x_end1, y_end1,RMSE_1,MAE_1,R2_1
            else:
                print(('RMSE :%.5f MAE: %.5f 1-R2: %.5f'%(RMSE_2,MAE_2,1-R2_2)))
                list_MAE.append(MAE_2)
                list_RMSE.append(RMSE_2)
                list_R2.append(R2_2)
                if R2_2>0.8:
                    list_hist_model.append(name_model)
                #return x_end2, y_end2,RMSE_2,MAE_2,R2_2
        if len(list_MAE)>0:
            data_out_1 = {'num':list_num,'name_model':list_name_model,'RMSE' : list_RMSE, 
                          'MAE' : list_MAE, 'R_squared': list_R2, }  # , 'Accuracy':list_Accuracy
            data_model_1 = pd.DataFrame(data_out_1)
            # _,_,_,_,data_model_2=nomalize_data_out(data_model_1)
            # name_excel_data_mean22 = 'E:/data_image_label_%s.xlsx' %name_model  
            #data_model_2.to_excel(name_excel_data_mean22)
            max_RMSE, max_MAE, max_R2,index_model,data_model_1,name_model_best  = nomalize_data_out(data_model_1)
            list_MAE_max.append(max_MAE)
            list_RMSE_max.append(max_RMSE)
            list_R2_max.append(max_R2)
            list_num_max.append(index_model)
            list_model.append(name_model_best)
            
        else: continue
    # labels, counts = np.unique(list_hist_model,return_counts=True) #gives you a histogram of your array 'a'
    # ticks = range(len(counts))
    # plt.bar(ticks,counts, align='center')
    # plt.xticks(ticks, labels)
    # plt.xticks(rotation=40)
    # plt.show()
    #print(max_R22)
    data_out = {'num':list_num_max,'name_model':list_model,'RMSE' : list_RMSE_max, 'MAE' : list_MAE_max, 'R_squared': list_R2_max, }  # , 'Accuracy':list_Accuracy
    data_model = pd.DataFrame(data_out)
    #print(data_model_1.max(axis=0))
    #print(data_model_1)
    max_RMSE, max_MAE, max_R2,index_model,data_model_1,name_model_best= nomalize_data_out(data_model)
    crack_predict,crack_nor = return_predict_crack_best(data_crack,name_model_best
                                              ,index_model,data_simulation,path_model,Snr_db)
    # name_excel_data_mean2 = 'E:/data_image_label3.xlsx' 
    # data_model_1.to_excel(name_excel_data_mean2)
    return  crack_nor,crack_predict,max_RMSE, max_MAE, max_R2,name_model_best    
        
    
if __name__ == '__main__':   
    #matplotlib.use('Agg')
    path_crack_simulation=r'D:\pix2pixHD\code\noise_cancellation_algorithm\data\Tex1000_1n_1cd__3p.txt'
    data = pd.read_csv(path_crack_simulation,
                    index_col=False,sep=",")
    data = (np.array(data))
    path_image_test ='D:/imgage_label/image_label.png'
    img = cv2.imread(path_image_test)
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
    path_model = 'D:\pix2pixHD\code\noise_cancellation_algorithm\model_linear\model\Tex1000_1n_1cd__3p'
    find_model_best_crack(data,point_crack,path_model)
    
    