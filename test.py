
# #test train model cancellation
# from NoiseCancellation.function_cancellation import *
# path_crack_simulation=r'D:\pix2pixHD\code\noise_cancellation_algorithm\data\Tex1000_1n_1cd__3p.txt'
# data = pd.read_csv(path_crack_simulation,
#                 index_col=False,sep=",")
# data = (np.array(data))
# path_image_test =r"D:\pix2pixHD\dataset\dataset_test\DCD\GT\11203.png"
# img = cv2.imread(path_image_test)
# #point_crack = np.argwhere(img == 255)
# val_resize = 1
# size_img = img.shape
# new_width = int(size_img[1]*val_resize)
# new_height = int(size_img[0]*val_resize)
# img_resized = cv2.resize(src=img, dsize=(new_width, new_height),interpolation = cv2.INTER_AREA)
# img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
# img_resized[np.argwhere(img_resized>0)[:,0],np.argwhere(img_resized>0)[:,1]] = 1
# skeleton = skeletonize(img_resized, method='lee') 
 
# point_crack = np.argwhere(skeleton > 0)
# if abs(len(np.unique(point_crack[:, 0])) - len(point_crack[:, 0])) > abs(len(np.unique(point_crack[:, 1])) - len(point_crack[:, 1])):
#     columnIndex = 1
#     # Sort 2D numpy array by 2nd Column
#     point_crack = point_crack[point_crack[:,
#                                         columnIndex].argsort()]
# else:
#     columnIndex = 0
#     # Sort 2D numpy array by 2nd Column
#     point_crack = point_crack[point_crack[:,
#                                         columnIndex].argsort()]

# path_model = r'D:\pix2pixHD\code\noise_cancellation_algorithm\model_linear\model\Tex1000_1n_1cd__3p'
# crack_nor,crack_predict,max_RMSE, max_MAE, max_R2,name_model_best =find_model_best_crack(data,point_crack,path_model,60,num_point_add=10)
# print(max_RMSE, max_MAE, max_R2,name_model_best)    
# plt.plot(crack_predict[:,1],crack_predict[:,0])
# plt.plot(crack_nor[:,1],crack_nor[:,0])
# plt.axis('equal')
# plt.show()    

from noise_cancellation.parameter_calculate_dataset import *
from noise_cancellation.train_noise_cancellation import *
from noise_cancellation.save_image_cancellation import *
from noise_cancellation.model_evaluation import *
from save_grap import *
#############################
# find SNR best in [["Crack_500","Cracktree","CrackForest","CRKWH_100","CrackLS315"]]
    # global MAE_data,RMSE_data, R2_data,M2_data,lst_SNR
    # MAE_data,RMSE_data, R2_data,M2_data,lst_SNR=[],[],[],[],[]
    # algos_data =  ["CFD","Crack_500","CrackLS315","CrackTree260","CRKWH100"]
    # processes = [multiprocessing.Process(target=find_SNR_process, args=[num_SNR,algos_data]) 
    #             for num_SNR in range(10,20,5)]
    # # start the processes
    # for process in processes:
    #     process.start()
    # # wait for completion
    # for process in processes:
    #     process.join()
    # df_RMSE=pd.DataFrame(data=MAE_data, index=lst_SNR, columns=algos_data)    
    # df_MAE=pd.DataFrame(data=RMSE_data,  index=lst_SNR,columns=algos_data)
    # df_R2=pd.DataFrame(data=R2_data,  index=lst_SNR,columns=algos_data)
    # df_M2=pd.DataFrame(data=M2_data, index=lst_SNR, columns=algos_data)
    # df_M2['sum'] =  df_M2.sum(axis=1)
   # SNR_best = df_M2['sum'].idxmin()
    # save_fig(df_RMSE,'RMSE')
    # save_fig(df_MAE,'MAE')
    # save_fig(df_R2,'R2')
    # save_fig(df_M2,'M2')
    
SNR_best = find_best_SNR_process()

# ##############################################################
#train model
from model_linear.train_model_linear_simulation import *
matplotlib.use('Agg')
list_class_model=[Linear_Model_LM,Gaussian_GPR,Tree,SVM]
show_fig= None
path_crack_simulation='./data/Tex1000_1n_1cd__3p.txt'
path_save ='./model_linear/model/'
save_model_ML = True
path_image_test ='D:/imgage_label/image_label.png'
Snr_db=SNR_best
num_point_add =10 
train_model_linear(list_class_model,path_crack_simulation,path_save,show_fig,save_model_ML,path_image_test,Snr_db,num_point_add)
######################################################################3
algos =  ["Deepcrack","EHCNN","FCN","HRNBM0.3","HED","Unet"]
###############################################################
list_test=["train","test"]
for typerun in list_test:
    total = 0
    #algos = ["Deepcrack","EHCNN","FCN","HRNBM0.3","HED","Unet"]
    save_folder_path = "./output/out_excel/%s_parameter"%(typerun) # Path to output folder
    #path GT
    gt_folder_path = './data/image_GT'
    # path image segmetation
    inputSeg_path = './data/image_Seg/{}'.format(typerun)
    # path data crack simulink
    path_crack_txt = './data/Tex1000_1n_1cd__3p.txt'
    #path model machine 
    path_model_folder_input = './model_linear/model/Tex1000_1n_1cd__3p_%s'%SNR_best
    # path orig image
    image_folder_path='./data/image/{}'.format(typerun)
    
    calculate_parameter_dataset(algos,save_folder_path,gt_folder_path,inputSeg_path,
            path_crack_txt,path_model_folder_input,image_folder_path)   

        
############################################################################
para_list=['EHR','EH','ER','HR']
for para_type in para_list:
    name_para=para_type
    path_save_model = './noise_cancellation/model_cancellation'
    seed = 42
    folds = 5
    #algos = ["Deepcrack","EHCNN","FCN","HRNBM0.3","HED","Unet"]
    path_parameter_train='./output/out_excel/train_parameter'
    train_model_cancellation(name_para,path_save_model,algos,seed,folds,path_parameter_train)

    ########################################################################################
    name_para=para_type
    #path_save_model = '/media/user1/Backup1/Hieu/NoiseCancellation/model/model_{}'.format(para_type)
    #total = 0
    #algos = ["Deepcrack","EHCNN","FCN","HRNBM0.3","HED","Unet"]
    image_folder_path='./data/image/test'
    inputSeg_path = "./data/image_Seg/test" #Path to seg
    list_image_input = glob.glob("{}/*.png".format('./data/image_Seg/test/Deepcrack'))
    path_parameter_test='./output/out_excel/test_parameter'
    #list_parameter=pd.read_excel(path_parameter_test)
    str_model=0
    save_image_cancellation(str_model,path_save_model,algos,name_para,
                            image_folder_path,inputSeg_path,list_image_input,path_parameter_test)
    
    ################################################################################
    #algos = ["Deepcrack","EHCNN","FCN","HRNBM0.3","HED","Unet"]
    list_image_in =glob.glob("{}/*.png".format('./data/image_Seg/test/Deepcrack'))
    seg_folder_path ='./output/image_out/image_out_{}'.format(para_type)
    path_GT = './data/image_GT'
    avg_list_precision,avg_list_Recall,avg_list_Accuracy,avg_list_F1=[],[],[],[]
    avg_list_precision,avg_list_Recall,avg_list_Accuracy,avg_list_F1=evaluation_dataset(
            algos,list_image_in,seg_folder_path,path_GT)
    print('-------------------------------------------------')
    date_object = datetime.date.today()
    path_1='./output/out_excel/'
    try:
        os.makedirs(path_1 +'_%s_'%(date_object))
        #os.makedirs(path_1 +'/_%s_/mean'%(date_object))
    except:
        print('done make folder') 
    path_save= path_1  +'/_%s_'%(date_object)  
    data_out_2 = {'name image':algos,'precision': avg_list_precision,'Recall': avg_list_Recall, 
                'Accuracy': avg_list_Accuracy, 'F1': avg_list_F1} 
    data_model_2 = pd.DataFrame(data_out_2)
    data_model_2.to_excel(path_save+'/data_mean_gt_%s.xlsx'%para_type)    


################################################################################
#algos = ["Deepcrack","EHCNN","FCN","HRNBM0.3"]
list_image_in =glob.glob("{}/*.png".format('./data/image_Seg/test/Deepcrack'))[1:2]
seg_folder_path ='./data/image_Seg/test'
path_GT = './data/image_GT'
avg_list_precision,avg_list_Recall,avg_list_Accuracy,avg_list_F1=evaluation_dataset(
        algos,list_image_in,seg_folder_path,path_GT)
print('-------------------------------------------------')
date_object = datetime.date.today()
path_1='./output/out_excel/'
try:
    os.makedirs(path_1 +'_%s_'%(date_object))
    #os.makedirs(path_1 +'/_%s_/mean'%(date_object))
except:
    print('done make folder') 
path_save= path_1  +'_%s_'%(date_object)  
data_out_2 = {'name image':algos,'precision': avg_list_precision,'Recall': avg_list_Recall, 
                'Accuracy': avg_list_Accuracy, 'F1': avg_list_F1} 
data_model_2 = pd.DataFrame(data_out_2)
data_model_2.to_excel(path_save+'/data_mean_gt.xlsx')  