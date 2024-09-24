from Utils.dataset import MyDataset
from Utils.SignalProcessing import *
from torchvision.utils import make_grid, save_image
import tqdm
from tensorboardX import SummaryWriter
from torch import optim
import torch.nn as nn
import torch
import time
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
from torch.autograd import Variable
from MTC import *
from Utils.utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr_gen", type=float, default=0.01, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_rate", type=float, default=0.98, help="lr decay rate")
parser.add_argument('--save_path', default=r'XXXXXXX', type=str, help='your save path')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:",device)

generator= Generator()
generator.to(device)
generator.apply(weights_init_normal)

train_path_x = "XXXXXX"     # your path
train_path_y = "XXXXXX"     # your path

full_dataset = MyDataset(train_path_x, train_path_y)
test_size = int(len(full_dataset) * 0.2)
train_size = len(full_dataset) - test_size

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset,[train_size, test_size])
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
optim_gen = optim.Adam(generator.parameters(), lr=args.lr_gen)

g_loss = nn.MSELoss()

writer = SummaryWriter()
writer_dict = {'writer':writer}
writer_dict["train_global_steps"]=0
temp_sets1 = []  
temp_sets2 = []  
temp_sets3 = []  
temp_sets4 = []  

start_time = time.strftime("1. %Y-%m-%d %H:%M:%S", time.localtime())  

for epoch in range(args.epochs):

    lr = args.lr_gen * (args.decay_rate ** epoch)
    for param_group in optim_gen.param_groups:
        param_group['lr'] = lr
    D_train_loss = 0.0
    G_train_loss = 0.0
    gen_step = 0

    generator = generator.train()
    
    for batch_idx1, (batch_x, batch_y) in enumerate(train_loader, 0):  
        global_steps = writer_dict['train_global_steps']

        batch_x = batch_x.to(device=device, dtype=torch.float32)
        batch_y = batch_y.to(device=device, dtype=torch.float32)
        # ---------
        #  Train 
        # ---------
        fake_img = generator(batch_x)
        G_loss = g_loss(fake_img,batch_y)
   
        G_train_loss += G_loss.item()
        optim_gen.zero_grad()    
        G_loss.backward()
        optim_gen.step()
        gen_step += 1

        if gen_step and batch_idx1 % 100 == 0:
            sample_imgs = fake_img[:25]
            img_grid = make_grid(sample_imgs, nrow=5, normalize=True, scale_each=True)
            gen_datas_array = sample_imgs.detach().cpu().numpy()
            directory = args.save_path + r'\save_gendata'   
            filename = f'./generated_datas_epoch{epoch}_batch{batch_idx1}.npz'  
            filepath = directory + '/' + filename
            np.savez(filepath, gen_datas_array )    
    G_train_loss = G_train_loss / (batch_idx1+1)  

    """
         valid
    """  
    generator = generator.eval()
    error_set = 0.0
    RMSE_set = 0.0
    R_square_set = 0.0
    G_val_loss = 0.0

    for batch_idx2, (val_x, val_y) in enumerate(test_loader, 0):
        val_x = val_x.to(device=device, dtype=torch.float32)
        val_y = val_y.to(device=device, dtype=torch.float32)

        with torch.no_grad(): 
            
            gen_data2 = generator(val_x)
            G_loss2 = g_loss(gen_data2, val_y)
            error = calculate_error(gen_data2, val_y)  
            RMSE = calculate_rmse(gen_data2, val_y)
            G_val_loss += G_loss2.item()  
        error_set += error 
        RMSE_set += RMSE
       
    error_set = error_set / (batch_idx2 + 1)
    RMSE_set = RMSE_set / (batch_idx2 + 1)

    G_val_loss = G_val_loss / (batch_idx2 + 1)
    loss_set = [G_train_loss, G_val_loss]

    temp_sets1.append(loss_set)
    temp_sets2.append(error_set)
    temp_sets3.append(RMSE_set)
    temp_sets4.append(R_square_set)
    print(
            "[Epoch %d/%d] [G_train_loss: %4f] [G_val_loss: %4f] [ERROR: %4f] [RMSE: %4f]"
            % (epoch, args.epochs, G_train_loss, G_val_loss, error_set, RMSE_set)
        )
    
    model_name = f'MTC_epoch{epoch+1}' 
    torch.save(generator, os.path.join(args.save_path + r'\save_MTC', model_name+'.pth'))  
end_time = time.strftime("1. %Y-%m-%d %H:%M:%S", time.localtime())  


np.savetxt(args.save_path + r'\save_MTC\\loss_sets.txt', temp_sets1, fmt='%.8f')
np.savetxt(args.save_path + r'\save_MTC\\error_sets.txt', temp_sets2, fmt='%.8f')
np.savetxt(args.save_path + r'\save_MTC\\RMSE_sets.txt', temp_sets3, fmt='%.8f')

data = np.loadtxt(args.save_path + r'\save_MTC\\loss_sets.txt')
data1 = np.loadtxt(args.save_path + r'\save_MTC\\error_sets.txt')
plt.figure()
plt.plot(data[:,0],label='G_train_loss')
plt.plot(data[:,1],label='G_val_loss')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.title("Loss")
plt.savefig(args.save_path + r'\save_MTC\\Loss.png')

plt.figure()
plt.plot(data1,label='error')
plt.xlabel("epoch")
plt.ylabel("error")
plt.legend()
plt.savefig(args.save_path + r'\save_MTC\\error.png')