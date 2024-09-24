import numpy as np
import matplotlib.pyplot as plt
from Utils.SignalProcessing import *
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_missing = np.load('model-main\PMTC\Test_data\input.npy')  
data_clean = np.load('model-main\PMTC\Test_data\label.npy') 

data_missing1 = data_missing
data_missing = np.expand_dims(np.expand_dims(data_missing, axis=0), axis=0)
data_missing = torch.from_numpy(data_missing)
data_missing = data_missing.to(device=device, dtype=torch.float32) 

model = torch.load('model-main\PMTC\Trained_model\\abalation\save_MTC\MTC_epoch200.pth')
model.to(device=device)  
model.eval()  

data_rec = model(data_missing)  
data_rec = data_rec.data.cpu().numpy()  
data_rec = data_rec.squeeze()  
data_missing = data_missing.data.cpu().numpy()  
data_missing = data_missing.squeeze()

data_rec = data_rec[9,:]
data_clean = data_clean[9,:]
error = error1(data_rec, data_clean)   
print("[error: %4f]"% (error))

plt.figure()
np.savetxt('MTC_abalation.txt',data_rec)
plt.plot(data_rec, label='data_rec') 
plt.plot(data_clean, label='data_clean')

plt.xlabel("Time")
plt.ylabel("Acceleration")
plt.legend()
plt.title("Result")
plt.show()
