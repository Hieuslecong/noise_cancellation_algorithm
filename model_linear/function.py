from model_linear.lib import *

##########################################################################################
 
def nomalize_data_crack(data,len_crack):
    if len_crack> 1:
        data_out = (data) /len_crack
        data_out =data_out - data_out.min()
        data_out = np.resize(data_out, ([np.size(data_out), 1]))
        return data_out+10
    else:
        valcrack=1/len_crack
        data_out = (data)*valcrack #/data.max()
        data_out =data_out - data_out.min()
        data_out = np.resize(data_out, ([np.size(data_out), 1]))
        return data_out+10
    
def nomalize_data_simulation(data,len_simulation):
    if len_simulation> 1:
        data_out = data /len_simulation
        data_out =data_out - data_out.min()
        data_out = np.resize(data_out, ([np.size(data_out), 1]))
        return data_out+10
    else:
        val_simulation=1/len_simulation
        data_out = data*val_simulation #/data.max()
        data_out =data_out - data_out.min()
        data_out = np.resize(data_out, ([np.size(data_out), 1]))
        return data_out+10
    
    
def nomalize_data_minmax(data):
    data_out = (data)/(data.max()-data.min()) 
    #data_out =data_out -data_out.mean()+1
    data_out = np.resize(data_out, ([np.size(data_out), 1]))
    return data_out
def nomalize_crack_simulation(data):
    #data_out = (data)/(data.max()-data.min()) 
    data_out =data -data.min()+1
    data_out = np.resize(data_out, ([np.size(data_out), 1]))
    return data_out
def nomalize_data_crack_to_simulation(data_simulation,data_crack):
    len_crack=np.sqrt((data_crack[0,0] - data_crack[-1,0])**2+(data_crack[0,1]-data_crack[-1,1])**2)
    len_simulation=np.sqrt((data_simulation[0,0] - data_simulation[-1,0])**2+(data_simulation[0,1]-data_simulation[-1,1])**2)
    ####### crack
    if len_crack==0:
        len_crack=len(data_crack[:,0])
    x_son = nomalize_data_crack(data_crack[:, 0],len_crack) #-min(nomalize_data_crack(data_crack[:, 0]))
    x_son = np.resize(x_son, ([np.size(x_son), 1]))
    y_son = nomalize_data_crack(data_crack[:, 1],len_crack) #-min(nomalize_data_crack(data_crack[:, 1]))
    y_son = np.resize(y_son, ([np.size(y_son), 1]))
    # crack simualation 
    x_simu =nomalize_data_simulation(data_simulation[:,0],len_simulation) #-min(data_simulation[:,0]*a)
    x_simu = np.resize(x_simu, ([np.size(x_simu), 1]))
    y_simu = nomalize_data_simulation(data_simulation[:,1],len_simulation)#-min(data_simulation[:,1]*a)
    y_simu = np.resize(y_simu, ([np.size(y_simu), 1]))
    # #
    # plt.plot(y_son,x_son)
    # #plt.plot(y-min(y),x-min(x))
    # plt.plot(x_simu,y_simu)
    # #plt.plot(y_out,x_out)
    # plt.axis('equal')
    # plt.show()
    #
    x_out,y_out,x_out2,y_out2=unify_data_crack(np.hstack((x_son,y_son)),np.hstack((y_simu,x_simu)))
    # plt.plot(y_son,x_son)
    # #plt.plot(y,x)
    # plt.plot(y_out,x_out)
    # plt.plot(x_simu,y_simu)
    # #plt.plot(y_out,x_out)
    # plt.axis('equal')
    # plt.show()
    #nor_crack_simulation=nomalize_data_minmax(data_simulation)
    #data_out= restore_data_crack(nor_crack_simulation,data_crack)
    x_out = np.resize(x_out, ([np.size(x_out), 1]))
    y_out = np.resize(y_out, ([np.size(y_out), 1]))
    x_out2 = np.resize(x_out2, ([np.size(x_out), 1]))
    y_out2 = np.resize(y_out2, ([np.size(y_out), 1]))
    return  x_out,y_out,x_out2,y_out2,x_simu,y_simu
    

def save_model(model,i,path_save,name_model,name_txt,num_SNR):
    name_txt =  name_txt + '_SNR_'+str(num_SNR)
    try:
        os.makedirs(path_save +name_txt)  
    except:
        print('done make dir') 
    try:
        os.makedirs(path_save + name_txt +'/' + name_model)
    except:
        print('done make dir') 
            
    if i<10:
        filename_model5 = path_save + '/%s/%s/sample_crack_000%d.model' %(name_txt,name_model,i)
    elif i>=10 and i<100:
        filename_model5 = path_save + '/%s/%s/sample_crack_00%d.model' %(name_txt,name_model,i)
    elif i>=100 and i<1000:
        filename_model5 = path_save + '/%s/%s/sample_crack_0%d.model' %(name_txt,name_model,i)
    else:
        filename_model5 = path_save + '/%s/%s/sample_crack_%d.model' %(name_txt,name_model,i)
    # save model 
    joblib.dump(model, filename_model5)

############################################# function #####################################################

def return_data_crack(point_crack_turn, point_crack_stard):
    point_crack1 = point_crack_turn
    point_crack2 = point_crack_stard
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
    if alpha >= alpha1:
        delta1 = alpha - alpha1
        delta2 = math.pi + delta1
    if alpha < alpha1:
        delta1 = math.pi - abs(alpha - alpha1)
        delta2 = math.pi + abs(delta1)
    x1 = x1 - x1[round(len(x1) / 2)]
    y1 = y1 - y1[round(len(y1) / 2)]
    x_end = x1 * math.cos(delta1) - y1 * math.sin(delta1) + x[round(len(x) / 2)]
    y_end = x1 * math.sin(delta1) + y1 * math.cos(delta1) + y[round(len(y) / 2)]
    x_end2 = x1 * math.cos(delta2) - y1 * math.sin(delta2) + x[round(len(x) / 2)]
    y_end2 = x1 * math.sin(delta2) + y1 * math.cos(delta2) + y[round(len(y) / 2)]
    # df_crack_trun = pd.DataFrame({'X_trun_1':x_end[:,np.newaxis],'y_trun_1':y_end[:,np.newaxis],'X_trun_2':x_end2[:,np.newaxis],'y_trun_2':y_end2[:,np.newaxis]})
    return x_end, y_end, x_end2, y_end2

#######################################################################################
