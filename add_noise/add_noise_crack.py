#import Library files numpy and matplot
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
# def find_linearx(x1,y1,x2,y2):
#     X= np.linspace(x1, x2, 100)
#     Y = (y2-y1)*(X -y1)/(x2-x1)+y1
#     return X,Y
def find_lineary(x1,y1,x2,y2):
    
    Y = np.linspace(y1, y2, 10)
    X= np.ones([10,1])*x1
    #Y = np.resize(Y, ([np.size(Y), 1]))
    X,Y=noise_1(X,Y)
    return X,Y
def find_linearxy(x1,y1,x2,y2):
    X= np.linspace(x1, x2, 10)
    Y = (y2-y1)*(X-x1)/(x2-x1)+y1
    X,Y=noise_1(X,Y)
    #Y = np.resize(Y, ([np.size(Y), 1]))
    return X,Y
def noise_1(t,signal):
    # t =x11
    # signal =y11
    # plt.plot(t, signal)
    # plt.title('signal')
    # plt.ylabel("Hagnitude")
    # plt.xlabel('Time (s)')
    # plt.show()

    ###########################
    power = signal ** 2
    # plt.plot(t, power)
    # plt.title("signal Power")
    # plt.ylabel( "Power (W)")
    # plt.xlabel ("Time (s)")
    # plt.show()

    signalpower_db = 10 *np.log10(power) #covert power in dB
    # plt.plot(t, signalpower_db)
    # plt.title('signal Power in dB')
    # plt.ylabel( "Power (48)")
    # plt.xlabel( "Time (s)")
    # plt.show()
    ########################################################
    Snr_db= 60 # add SWR of 20 4B
    import math


    signal_average_power = np.mean(power) # Calculate signal power
    signal_averagepower_db = 10 * math.log10(signal_average_power) 
    print(signal_averagepower_db)#convert signal power to 48
    noise_db = signal_averagepower_db - Snr_db # Calculate noise
    noise_watts = 10 ** ( noise_db/10) # convert noise from db to watts
    # Generate an sample of white noise
    mean_noise =0
    noise = np.random.normal(mean_noise, np.sqrt(noise_watts), len(signal))
    noise_signal = signal + noise # Noise added to the original 

    #########################################
    # Plot noise signal
    # plt.plot(t, noise_signal)
    # plt.title("signal with noise")
    # plt.ylabel("Magnitude")
    # plt.xlabel("Tine (s)")
    # plt.show()
    return t,noise_signal

    #########################################################
    
        

data = pd.read_csv("D:\crack_simulation\Tex1000_1n_1cd__3p.txt",
                    index_col=False,sep=",")
data = (np.array(data))
cloumn1 = data[:, 0]
point = np.argwhere(cloumn1 == 0)[:, 0]
point = np.append([-1], point)
#print(len(point))
fig, ax = plt.subplots(figsize=(6, 6))
for i in range(len(point)-1):
    #print(i)
    data1 = data[point[i] + 1:point[i + 1], :]
    X = data1[:, 0] - (data1[:, 0]).mean() + 1
    X = np.resize(X, ([np.size(X), 1]))
    y = data1[:, 1] - (data1[:, 1]).mean() + 1
    y = np.resize(y, ([np.size(y), 1]))
    t =X
    signal =y
    plt.plot(t, signal)
    
    #print(X,y)
    x11=[]
    y11=[]
    for k in range(0,len(X)-1):
        print(k)
        x1=X[k,0]
        x2=X[k+1,0]
        y1=y[k,0]
        y2=y[k+1,0]
        print(x1,x2,x2-x1)
        if x2-x1 ==0:
            X1,Y1= find_lineary(x1,y1,x2,y2)
        else:
            X1,Y1= find_linearxy(x1,y1,x2,y2)
        x11=np.append(x11, X1)
        y11=np.append(y11, Y1)
        y11 = np.resize(y11, ([np.size(y11), 1]))
        print(X1,Y1)
        
    plt.plot(x11+0.02, y11)
    plt.title("signal with noise")
    plt.ylabel("Magnitude")
    plt.xlabel("Tine (s)")
    plt.show()  
    #t= np.linspace(1, 100, 1000) # Generate 1000 samples fron 1 to 100
    #signal = 10*np.sin(t/(2*np.pi))

