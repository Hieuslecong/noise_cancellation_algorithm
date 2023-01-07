from add_noise.lib import *

def noise_1(t,signal,Snr_db=60):
        power = signal ** 2
        signalpower_db = 10 *np.log10(power) #covert power in dB
        #Snr_db= 60 # add SWR of 20 4B
        signal_average_power = np.mean(power) # Calculate signal power
        signal_averagepower_db = 10 * math.log10(signal_average_power) 
        noise_db = signal_averagepower_db - Snr_db # Calculate noise
        noise_watts = 10 ** (noise_db/10) # convert noise from db to watts
        # Generate an sample of white noise
        mean_noise =0
        noise = np.random.normal(mean_noise, np.sqrt(noise_watts), len(signal))
        noise_signal = signal + noise # Noise added to the original 
        return t,noise_signal
    
class Gaussian_noise:
    def find_lineary(x1,y1,x2,y2,Snr_db=60,num_point=10):
        Y = np.linspace(y1, y2, num_point)
        X= np.ones([num_point,1])*x1
        #Y = np.resize(Y, ([np.size(Y), 1]))
        X,Y =noise_1(X,Y,Snr_db)
        return X,Y
    def find_linearxy(x1,y1,x2,y2,Snr_db=60,num_point=10):
        X= np.linspace(x1, x2, num_point)
        Y = (y2-y1)*(X-x1)/(x2-x1)+y1
        X,Y =noise_1(X,Y,Snr_db)
        #Y = np.resize(Y, ([np.size(Y), 1]))
        return X,Y