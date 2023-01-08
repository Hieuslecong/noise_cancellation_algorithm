
from noise_cancellation.lib import *
# train_model_crack = argparse.ArgumentParser(description='Image Classification Using PyTorch', usage='[option] model_name')
# train_model_crack.add_argument('--name_output', type=str, required=True)
# args = train_model_crack.parse_args()
def save_image_cancellation(str_model,path_save_model,algos,name_para,
                            image_folder_path,inputSeg_path,list_image_input,path_parameter):

    """
    --------Change paths to GT,output and segmentation folder
    """
    save_folder_path = "./output/image_area" # Path to output folder
    save_folder_path1 = "./output/image_out/image_out_{}".format(name_para)
    #inputSeg_path = "/Users/doanquangmanh/Desktop/Manhtest/Research/GAPs" #Path to seg
    if not os.path.exists(save_folder_path):
        os.mkdir(save_folder_path)
    if not os.path.exists(save_folder_path1):
        os.mkdir(save_folder_path1)
    model_path = "{}/{}/Model_balance{}".format(path_save_model,name_para,str_model) # Path to model file
    norm_path = "{}/{}/Normalization_balance{}".format(path_save_model,name_para,str_model) # Path to normalization file
    # normalization file
    print(model_path)
    mo = pickle.load(open(model_path,'rb'))
    no = pickle.load(open(norm_path,'rb')) 
    
    """
    END
    """
    
    # image_paths = glob.glob("{}/*.png".format(gt_folder_path))
    for algo in algos:
        seg_folder_path = "{}/{}".format(inputSeg_path,algo) # Path to seg
        output_subfolder = "{}/{}".format(save_folder_path1,algo)
        if not os.path.exists(output_subfolder):
            os.mkdir(output_subfolder)
        arr = []
        NUM=0
        image_paths = glob.glob("{}/*.png".format(seg_folder_path))
        list_parameter = pd.read_excel('{}/{}_labeled.xlsx'.format(path_parameter,'data'))
        for f,path in enumerate(list_image_input):
            print(f+1,"/",len(list_image_input))
    # path = "/Users/doanquangmanh/Desktop/Manhtest/Research/GAPs/Deepcrack/GAPS384_test_0028_541_1.png"
            name = (os.path.split(path)[-1]).split(".")[0]
            # I = cv2.imread(path)
            seg = cv2.imread("{}/{}.png".format(seg_folder_path,name))
            Orig = cv2.imread("{}/{}.jpg".format(image_folder_path,name))
            # I = cv2.resize(I,seg.shape[:2])
            if len(seg.shape)>2:
                seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
            if len(Orig.shape)>2:
                Orig = cv2.cvtColor(Orig, cv2.COLOR_BGR2GRAY)
            # orig = I.copy()
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
                
                if x_min > 10:
                    x_min -= 10
                if y_min > 10:
                    y_min -= 10
                if x_max < seg.shape[0]-10:
                    x_max += 10
                if y_max < seg.shape[1]-10:
                    y_max += 10
                region_seg = SEG[x_min:x_max+1,y_min:y_max+1]
                region_Orig = Orig[x_min:x_max+1,y_min:y_max+1]
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
                #fdmean = (1000*fdmean)**6
                
                cv2.imwrite("{}/{}_region_{}.png".format(save_folder_path,name,k), region_seg)
                img=cv2.imread("{}/{}_region_{}.png".format(save_folder_path,name,k))
                #img = cv2.cvtColor(skeleton, cv2.COLOR_BGR2GRAY)
                    #img = region_seg
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    #cv2.imwrite("{}/{}_region_{}.png".format(output_subfolder,name,k), region_seg)
                skeleton = skeletonize(img, method='lee')
                point_crack = np.argwhere(skeleton > 0)
                
                if numpxs_seg < 3 or len(point_crack[:, 0]) < 3:
                    RMSE, MAE, R2 = 0,0,0
                    NUM=NUM+1
                else:
                    """
                    ------- Calculate RMSE,MAE,R2 ----------
                    """
                    print(NUM)
                    RMSE = list_parameter.iloc[NUM,5]
                    NUM=NUM+1
                    """
                    ------- END ----------------------------
                    """
                # cv2.imwrite("{}/{}_region_{}.png".format(output_subfolder,name,k), region_seg)
                if name_para =='EHR':
                    X_test = no.transform(np.array([[E,fdmean,RMSE]]))
                    y_pred = mo.predict(X_test)
                elif name_para =='HR':
                    X_test = no.transform(np.array([[fdmean,RMSE]]))
                    y_pred = mo.predict(X_test)
                elif name_para =='ER':
                    X_test = no.transform(np.array([[E,RMSE]]))
                    y_pred = mo.predict(X_test)
                else:
                    X_test = no.transform(np.array([[E,fdmean]]))
                    y_pred = mo.predict(X_test)
                if y_pred == 0:
                    seg[x,y] = 0
            cv2.imwrite("{}/{}.png".format(output_subfolder,name), seg)
        
        
    #     cv2.imshow("region",region_seg)
    #     cv2.waitKey(0)
    # cv2.imshow("final",seg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
if __name__ == '__main__': 
    name_para='test'
    path_save_model = '/media/user1/Backup1/Hieu/NoiseCancellation/model/model_EHR_{}'.format(name_para)
    total = 0
    algos = ["Deepcrack","EHCNN","FCN","HRNBM0.3","HED","Unet"]
    image_folder_path='/media/user1/Backup1/Hieu/cvpr/GAPs/copy/GAPS384'
    inputSeg_path = "/media/user1/Backup1/Hieu/NoiseCancellation/data/test" #Path to seg
    list_image_input = glob.glob("{}/*.png".format("/media/user1/Backup1/Hieu/NoiseCancellation/data/test/Deepcrack"))
    list_parameter=pd.read_excel("/media/user1/Backup1/Hieu/cvpr/data_EHR_test1/EHR_test_labeled.xlsx")
    k=0
    save_image_cancellation(k,path_save_model,algos,name_para,image_folder_path,inputSeg_path,list_image_input,list_parameter)