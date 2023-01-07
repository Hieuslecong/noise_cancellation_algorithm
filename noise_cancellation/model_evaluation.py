from noise_cancellation.lib import *
def get_iou(y_pred, y_true):
    # y_pred : Segmentation
    # y_true: GT
    A = y_true
    B = y_pred
    intersection = np.logical_and(A, B)
    union = np.logical_or(A, B)
    iou = np.sum(intersection > 0) / np.sum(union > 0)

    return iou


def Accuracy1(GT,seg,beta):
    r = []
    p = []
    F = 0
    #[x,y] = np.argwhere(GT >0)
    GT[np.argwhere(GT >0)] = 1
    GT = np.ndarray.flatten(GT)
    #[x,y] = np.argwhere(seg > 0)
    seg[np.argwhere(seg > 0)] = 1
    seg = np.ndarray.flatten(seg)
    CM = confusion_matrix(GT,seg)
    c = np.shape(CM)
    for i in range(c[1]):
        if (np.sum(CM[i,:]) == 0):
            r.append(0)
        else:
            a = CM[i,i]/(np.sum(CM[i,:]))
            r.append(a)
        if (np.sum(CM[:,i]) == 0):
            p.append(0)
        else:
            p.append(CM[i,i]/(np.sum(CM[:,i])))
    F = (1+beta)*(np.mean(r)*np.mean(p))/(beta*np.mean(p)+np.mean(r))
    return F,np.mean(p),np.mean(r)

def evaluation_dataset(algos,list_image_in,seg_folder_path,path_GT):
    list_image,list_precision,list_Recall,list_Accuracy,list_F1=[],[],[],[],[]
    avg_list_image,avg_list_precision,avg_list_Recall,avg_list_Accuracy,avg_list_F1=[],[],[],[],[]
    #avg_list_image1,avg_list_precision1,avg_list_Recall1,avg_list_Accuracy1,avg_list_F11=[],[],[],[],[]
    for algo in algos:
        for path in list_image_in:
            name = (os.path.split(path)[-1]).split(".")[0]
            print(name)
            #I = cv2.imread("{}/{}.jpg".format(image_folder_path,name))
            #seg = cv2.imread("{}/{}/{}.png".format('/media/user1/Backup1/Hieu/cvpr/GAPs',list_2[i],name))
            seg = cv2.imread("{}/{}/{}.png".format(seg_folder_path,algo,name))
            img_gt = cv2.imread("{}/{}.png".format(path_GT,name))
            print("{}/{}/{}.png".format(seg_folder_path,algo,name))
            img_predict = seg
            if len(img_gt.shape) >2:
                img_gt = cv2.cvtColor(img_gt,cv2.COLOR_BGR2GRAY)
            if len(img_predict.shape) >2:
                img_predict = cv2.cvtColor(img_predict,cv2.COLOR_BGR2GRAY)
                seg = cv2.cvtColor(seg,cv2.COLOR_BGR2GRAY)
            img_gt[img_gt>0]=2
            seg[seg>0]=2
            #img_gt[img_gt==255]=0
            img_predict[img_predict>0]=2
            img_gt=img_gt.reshape(-1)
            img_predict=img_predict.reshape(-1)
            seg=seg.reshape(-1)
            print('-------------------------------------------------')
            # Tính accuracy: (tp + tn) / (p + n)
            f1,precision,recall=Accuracy1(img_gt,img_predict,0.3)
            accuracy = accuracy_score(img_gt, img_predict)
            # f11,precision1,recall1=Accuracy1(img_gt,seg1,0.3)
            # accuracy1 = accuracy_score(img_gt, seg1)
            #print('Accuracy: %f' % accuracy)
            # Tính precision tp / (tp + fp)
            #precision = precision_score(img_gt, img_predict, average='macro')
            #print('Precision: %f' % precision)
            # Tính recall: tp / (tp + fn)
            #recall = recall_score(img_gt, img_predict, average='macro')
            #print('Recall: %f' % recall)
            # Tính f1: 2 tp / (2 tp + fp + fn)
            #f1 = f1_score(img_gt, img_predict, average='macro')
        #noisecancellation.py print('F1 score: %f' % f1)
            list_precision.append(precision)
            list_Recall.append(recall)
            list_Accuracy.append(accuracy)
            list_F1.append(f1)
            
        avg_list_precision.append(sum(list_precision)/len(list_precision))
        avg_list_Recall.append(sum(list_Recall)/len(list_Recall))
        avg_list_Accuracy.append(sum(list_Accuracy)/len(list_Accuracy))
        avg_list_F1.append(sum(list_F1)/len(list_F1))
        print(avg_list_precision,sum(list_precision)/len(list_precision))
        print(avg_list_Recall,sum(list_Recall)/len(list_Recall))
        print(avg_list_F1,sum(list_F1)/len(list_F1))
        #list_precision,list_Recall,list_Accuracy,list_F1
    #data_out_1 = {'name image':image_paths,'precision': list_precision,'Recall': list_Recall, 'Accuracy': list_Accuracy, 'F1': list_F1, }
    
    #data_out_1 = {'name image':image_paths,'precision': list_precision,'Recall': list_Recall, 'Accuracy': list_Accuracy, 'F1': list_F1, }
    return avg_list_precision,avg_list_Recall,avg_list_Accuracy,avg_list_F1


if __name__ == '__main__': 
    algos = ["Deepcrack","EHCNN","FCN","HRNBM0.3","HED","Unet"]
    list_image_in =glob.glob("{}/*.png".format('/media/user1/Backup1/Hieu/NoiseCancellation/data/test/Deepcrack'))
    seg_folder_path ='/media/user1/Backup1/Hieu/cvpr/output_RMR'
    path_GT = '/media/user1/Backup1/Hieu/cvpr/img_resize'
    avg_list_precision,avg_list_Recall,avg_list_Accuracy,avg_list_F1=evaluation_dataset(
        algos,list_image_in,seg_folder_path,path_GT)
    print('-------------------------------------------------')
    date_object = datetime.date.today()
    path_1=''
    try:
        os.makedirs(path_1 +'/_%s_'%(date_object))
        os.makedirs(path_1 +'/_%s_/mean'%(date_object))
    except:
        print('done make folder') 
    path_save= path_1  +'_%s_'%(date_object)  
    data_out_2 = {'name image':algos,'precision': avg_list_precision,'Recall': avg_list_Recall, 'Accuracy': avg_list_Accuracy, 'F1': avg_list_F1, 'precision1': avg_list_precision1,'Recal1l': avg_list_Recall1, 'Accuracy1': avg_list_Accuracy1, 'F11': avg_list_F11 } 
    data_model_2 = pd.DataFrame(data_out_2)
    data_model_2.to_excel(path_save+'/mean/data_mean_gt_RMR.xlsx')