#import Library files numpy and matplot
from add_noise.lib import *
from add_noise.noise_model import *
import argparse
import matplotlib
matplotlib.use('Agg')
def add_noise(X_in,y_in,Snr_db,num_point):
    # cloumn1 = data[:, 0]
    # point = np.argwhere(cloumn1 == 0)[:, 0]
    # print(len(point))
    # point = np.append([-1], point)
    # #print(len(point))
    
    # for i in range(len(point)-1):
    #     #print(i)
    #     data1 = data[point[i] + 1:point[i + 1], :]
    #     X_in = data1[:, 0] - (data1[:, 0]).mean() + 1
    #     X_in = np.resize(X_in, ([np.size(X_in), 1]))
    #     y_in = data1[:, 1] - (data1[:, 1]).mean() + 1
    #     y_in = np.resize(y_in, ([np.size(y_in), 1]))
    t =X_in
    signal =y_in
    #plt.plot(t, signal)
    X_out=[]
    Y_out=[]
    show_grap=True
    for k in range(0,len(X_in)-1):
        point1_x=X_in[k,0]
        point1_y=y_in[k,0]
        point2_x=X_in[k+1,0]
        point2_y=y_in[k+1,0]
        if point2_x-point1_x ==0:
            X_noise,Y_noise= Gaussian_noise.find_lineary(point1_x,point1_y,point2_x,point2_y,Snr_db,num_point)
        else:
            X_noise,Y_noise= Gaussian_noise.find_linearxy(point1_x,point1_y,point2_x,point2_y,Snr_db,num_point)
        X_out=np.append(X_out, X_noise)
        Y_out=np.append(Y_out, Y_noise)
        Y_out = np.resize(Y_out, ([np.size(Y_out), 1]))
    
    if show_grap is not None:  
        # fig, ax = plt.subplots(figsize=(6, 6))
        plt.figure(figsize=(6, 6))
        plt.plot(X_out+ 0.05, Y_out)
        plt.plot(X_in, y_in)
        plt.title("signal with noise")
        plt.ylabel("Magnitude")
        plt.xlabel("Tine (s)")
        path = './output/save_grap/'
        if not os.path.exists(path):
                os.mkdir(path)
        path_save = path + 'exp_' + str(Snr_db)+ '_.png'
        #plt.savefig(path_save)
        plt.figure().clear()
        plt.close()
        plt.close('all')
        plt.cla()
        plt.clf()
        #plt.show()  
    #t= np.linspace(1, 100, 1000) # Generate 1000 samples fron 1 to 100
    #signal = 10*np.sin(t/(2*np.pi))
    X_out = np.resize(X_out, ([np.size(X_out), 1]))
    Y_out = np.resize(Y_out, ([np.size(Y_out), 1]))
    return   X_out, Y_out  

def main():
    add_noise_crack = argparse.ArgumentParser(description='add noise in crack ', usage='[option] model_name')
    add_noise_crack.add_argument('--path_data_simul', type=str,default='./data/Tex1000_1n_1cd__3p.txt', required=True)
    add_noise_crack.add_argument('--Snr_db', type=int, default=60,required=True)
    add_noise_crack.add_argument('--num_point', type=int, default=60,required=True)
    args = add_noise_crack.parse_args()
    path_data = args.path_data_simul
    data = pd.read_csv( path_data, index_col=False,sep=",")
    data = (np.array(data))
    add_noise(data,args.Snr_db,args.num_point)
    


if __name__ == '__main__':   
    main()
    
