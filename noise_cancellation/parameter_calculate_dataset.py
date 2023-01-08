from .lib import *
from .function_cancellation import *
#data set
def calculate_parameter_dataset(algos,save_folder_path,gt_folder_path,inputSeg_path,
         path_crack_txt,path_model_folder_input,image_folder_path):
    """
    --------Change paths to GT,output and segmentation folder
    """
    if not os.path.exists(save_folder_path):
        os.mkdir(save_folder_path)
    path_save_image_area='./output/image_area'
    if not os.path.exists(path_save_image_area):
        os.mkdir(path_save_image_area)
    """
    END
    """
    img_txt=[]
    num=0
    list_img=[]
    image_paths = glob.glob("{}/*.jpg".format(image_folder_path))[1:5]
    #path_crack_simulation=r'D:\pix2pixHD\code\noise_cancellation_algorithm\data\Tex1000_1n_1cd__3p.txt'
    data_crack_simulation = pd.read_csv(path_crack_txt,
                    index_col=False,sep=",")
    data_crack_simulation = (np.array(data_crack_simulation))
    arr = []
    for algo in algos:
        seg_folder_path = "{}/{}".format(inputSeg_path,algo) # Path to seg
        #output_subfolder = "{}/{}".format(save_folder_path,algo)
        output_save_image = "{}/{}".format(path_save_image_area,algo)
        # if not os.path.exists(output_subfolder):
        #     os.mkdir(output_subfolder)
        if not os.path.exists(output_save_image):
            os.mkdir(output_save_image)
        
        for f,path in enumerate(image_paths):
            print(f+1,"/",len(image_paths)," - ",algo)
            name = (os.path.split(path)[-1]).split(".")[0]
            seg = cv2.imread("{}/{}.png".format(seg_folder_path,name))
            I = cv2.imread("{}/{}.png".format(gt_folder_path,name))
            I = cv2.resize(I,seg.shape[:2])
            Orig = cv2.imread("{}/{}.jpg".format(image_folder_path,name))
            Orig = cv2.resize(Orig,seg.shape[:2])
            if len(seg.shape)>2:
                seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
            if len(I.shape)>2:
                I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
            if len(Orig.shape)>2:
                Orig = cv2.cvtColor(Orig, cv2.COLOR_BGR2GRAY)  
                
            orig = I.copy()
            labeled_seg = label(seg)
            data = regionprops_table(
                    labeled_seg,
                    properties=('label', 'eccentricity'),
                    )
            table = pd.DataFrame(data)
            Ecc = table.iloc[:,1].values
            numclasses = len(np.unique(labeled_seg))
            n_region = 0
            for k in range(1,numclasses):
                x,y = np.where(labeled_seg==k)[0],np.where(labeled_seg==k)[1]
                x_min = min(x)
                y_min = min(y)
                x_max = max(x)
                y_max = max(y)
                SEG = np.zeros(seg.shape)
                SEG[x,y] = seg[x,y]
                O = np.zeros(seg.shape)
                O[x,y] = I[x,y]
                
                if x_min > 10:
                    x_min -= 10
                if y_min > 10:
                    y_min -= 10
                if x_max < I.shape[0]-10:
                    x_max += 10
                if y_max < I.shape[1]-10:
                    y_max += 10
                region_I = O[x_min:x_max+1,y_min:y_max+1]
                region_seg = SEG[x_min:x_max+1,y_min:y_max+1]
                region_Orig = Orig[x_min:x_max+1,y_min:y_max+1]
                numpxs_I = len(np.where(region_I==255)[0])
                numpxs_seg = len(np.where(region_seg==255)[0])
                A = numpxs_seg
                E = Ecc[k-1]
                fd = hog(region_Orig, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=False, multichannel=False)
                if fd.size==0:
                    fdmax = 0
                    fdmean = 0
                else:
                    fdmax = np.max(fd)
                    fdmean = np.mean(fd)
                cv2.imwrite("{}/{}_region_{}.png".format(output_save_image,name,k), region_seg)
                list_img.append(path)
                if numpxs_I>0 and numpxs_seg>0:
                    la = 1
                else: 
                    la = 0
                img=cv2.imread("{}/{}_region_{}.png".format(output_save_image,name,k))
                skeleton = skeletonize(img, method='lee')
                num = 0
                point_crack = np.argwhere(skeleton > 0)
                
                if numpxs_seg < 3 or len(point_crack[:, 0]) < 3: 
                    max_RMSE, max_MAE, max_R2 = 0,0,0
                    #arr.append([list_img,A,E,fdmax,fdmean,RMSE,MAE,R2,la])
                else:
                    """
                    ------- Calculate RMSE,MAE,R2 ----------
                    """
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
                    crack_nor,crack_predict,max_RMSE, max_MAE, max_R2,name_model_best =find_model_best_crack(
                        data_crack_simulation,point_crack,path_model_folder_input,60,num_point_add=10)
                    #point_crack_train,max_RMSE, max_MAE, max_R2 = best_model_crack(point_crack)
                    fig, ax = plt.subplots(figsize=(6, 6))
                    #x12, y12,RMSE ,MAE, R2 = return_data_crack(point_crack_train, point_crack)
                    print(name_model_best)
                    print(('RMSE :%s MAE: %s 1-R2: %s'%(max_RMSE,max_MAE,1-max_R2)))
                    #RMSE, MAE, R2 = -1,-1,-1
                    #arr.append([list_img,A,E,fdmax,fdmean,RMSE,MAE,R2,la])
                    """
                    ------- END ----------------------------
                    """
                arr.append([list_img,A,E,fdmax,fdmean,max_RMSE,max_MAE,max_R2,la])
        #arr = np.array(arr)
    df = pd.DataFrame(arr)
    df.to_excel("{}/{}_labeled.xlsx".format(save_folder_path,'data'))
if __name__ == '__main__':   
    total = 0
    algos = ["Deepcrack","EHCNN","FCN","HRNBM0.3","HED","Unet"]
    save_folder_path = r"D:\pix2pixHD\code\noise_cancellation_algorithm\output\image_area" # Path to output folder
    #path GT
    gt_folder_path = r'D:\pix2pixHD\code\noise_cancellation_algorithm\NoiseCancellation\data\GAPs\img_resize'
    # path image segmetation
    inputSeg_path = r"D:\pix2pixHD\code\noise_cancellation_algorithm\NoiseCancellation\data\GAPs" 
    # path data crack simulink
    path_crack_txt = r'D:\pix2pixHD\code\noise_cancellation_algorithm\data\Tex1000_1n_1cd__3p.txt'
    #path model machine 
    path_model_folder_input = r'D:\pix2pixHD\code\noise_cancellation_algorithm\model_linear\model\Tex1000_1n_1cd__3p'
    # path orig image
    image_folder_path=r'D:\pix2pixHD\code\noise_cancellation_algorithm\NoiseCancellation\data\GAPs\copy\GAPS384'
    calculate_parameter_dataset(algos,save_folder_path,gt_folder_path,inputSeg_path,
         path_crack_txt,path_model_folder_input,image_folder_path)            
