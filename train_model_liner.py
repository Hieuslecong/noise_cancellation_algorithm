#train model
from model_linear.train_model_linear_simulation import *
matplotlib.use('Agg')
list_class_model=[Linear_Model_LM,Gaussian_GPR,Tree,SVM]
show_fig= None
path_crack_simulation=r'D:\pix2pixHD\code\noise_cancellation_algorithm\data\Tex1000_1n_1cd__3p.txt'
path_save =r'D:\pix2pixHD\code\noise_cancellation_algorithm\model_linear\model/'
save_model_ML = True
path_image_test ='D:/imgage_label/image_label.png'
Snr_db=60
num_point_add =10 
train_model_linear(list_class_model,path_crack_simulation,path_save,show_fig,save_model_ML,path_image_test,Snr_db,num_point_add)

