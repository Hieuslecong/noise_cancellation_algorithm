
from model_linear.lib import *
from model_linear.Model_machine import *
from model_linear.function import *
from add_noise.noise_crack import *

    
def train_model_linear(list_class_model,path_crack_simulation,path_save,show_fig,save_model_ML,path_image_test,Snr_db,num_point_add):
    #global i,R_squared
    lst_model=[]
    name_simuation =  (os.path.split(path_crack_simulation)[-1]).split(".")[0]
    # load data
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

    x_son = (point_crack[:, 0])
    x_son = np.resize(x_son, ([np.size(x_son), 1]))
    y_son = (point_crack[:, 1])
    y_son = np.resize(y_son, ([np.size(y_son), 1]))

    # dataframe test model
    # df_test = pd.DataFrame(
    #     {'test_x': x_son,'test_y': y_son})

    # load data crack train
    # data = pd.read_csv("F:/crack-machine/data_crack.txt",
    #                    delimiter='\s+', index_col=False)
    data = pd.read_csv(path_crack_simulation,
                    index_col=False,sep=",")
    data = (np.array(data))
    cloumn1 = data[:, 0]
    cloumn1 = np.resize(cloumn1, ([np.size(cloumn1), 1]))
    cloumn2 = data[:, 1]
    cloumn2 = np.resize(cloumn2, ([np.size(cloumn1), 1]))
    point = np.argwhere(cloumn1 == 0)[:, 0]
    point = np.append([-1], point)  # add the first crack loction

    # careate a list of crack names
    number_crack = []
    for i in range(0, len(point) - 1):
        number_crack_name = 'num_crack_%s' % i
        number_crack.append(number_crack_name)

    # load model
    for class_model in list_class_model:
        print(class_model)
        #call and run all model in foulder
        for name,method in call_all_method(class_model).items():
            print("calling: {}".format(name))
            try:
                name_model,model=method()
                lst_model.append((name_model,model))
            except:
                continue

    # train model
    list_name_model =[]
    out_data_model_mean = pd.DataFrame()
    
    for name_model, model in lst_model:
        
        #print(name_model)
        list_name_model.append(name_model)
        list_MSE, list_MSA, list_RMSE, list_R_squared, list_Accuracy = [], [], [], [], []
        list_MSE_mean, list_MAE_mean, list_RMSE_mean, list_R_squared_mean, list_Accuracy = [], [], [], [], [] 
        i = 0
        
        for i in range(0, len(point) - 1):
            data1 = ((data[point[i] + 1:point[i + 1], :]))+1
            
            #X_crack,y_crack,x_simu,y_simu=nomalize_data_crack_to_simulation(data1,point_crack)
            X = nomalize_crack_simulation(data1[:, 0])
            y = nomalize_crack_simulation(data1[:, 1] )
            # add noise
            x_simu,y_simu=add_noise(X,y,Snr_db=Snr_db,num_point=num_point_add)
            X_crack,y_crack,X_crack2,y_crack2,x_simu,y_simu=nomalize_data_crack_to_simulation(np.hstack((x_simu,y_simu)),point_crack)
            #x_simu,y_simu=add_noise(x_simu,y_simu,Snr_db=60,num_point=50)
            # train model in list
            try:
                model.fit(y_simu,x_simu.ravel())
            except:
                print(name_model)
            MSA = mean_absolute_error(y_crack, model.predict(X_crack))
            # MSE
            MSE = mean_squared_error(y_crack, model.predict(X_crack))
            # RMSE
            RMSE = np.sqrt(MSE)
            # R_squared
            R_squared = r2_score(y_crack, model.predict(X_crack))
            # accuracy
            print('%s crack number %s R_squared: %s MSA: %s MSE: %s' %
                (name_model, i, R_squared, MSA, MSE))
            #print(R_squared )
            # ani = animation.FuncAnimation(fig1, animate, interval=1000)
            # plt.show()
            # save model 
            if save_model_ML is not None:
                save_model(model,i,path_save= path_save,name_model=name_model,name_txt=name_simuation)
            if show_fig is not None:
                fig, ax = plt.subplots(figsize=(6, 6)) # Defines ax variable by creating an empty plot
                # Define the data to be plotted
                #plt.plot(x, y, 'b+', label='Data points')
                plt.scatter(y_crack,X_crack,
                            edgecolor='c', s=1, label="True crack")
                #plt.scatter((model.predict(x_son)), (x_son),
                #            edgecolor='c', s=1, label="True crack")
                plt.scatter(model.predict(X_crack),X_crack,  
                            edgecolor='k', s=1, label="True crack")
                plt.scatter(x_simu,y_simu, edgecolor='b', s=1, label="Crack samples %s" %i)
                #plt.scatter(y_son1, x_son1, edgecolor='r', s=0.51, label="Model train")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.axis('equal')
                # plt.xlim((-2, 2))
                # plt.ylim((-2, 2))
                #fig.savefig('filename.png', dpi=125)
                plt.show()
    # test model

if __name__ == '__main__':   
    matplotlib.use('Agg')
    
    list_class_model=[Linear_Model_LM,Gaussian_GPR,Tree,SVM]
    show_fig= True
    path_crack_simulation=r'D:\pix2pixHD\code\noise_cancellation_algorithm\data\Tex1000_1n_1cd__3p.txt'
    path_save = './model_linear/model/'
    save_model_ML = True
    path_image_test ='D:/imgage_label/image_label.png'
    Snr_db=60
    num_point_add =10
    train_model_linear(list_class_model,path_crack_simulation,path_save,show_fig,save_model_ML,path_image_test,Snr_db,num_point_add)
    
    
