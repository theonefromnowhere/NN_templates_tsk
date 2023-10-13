


import os


import pytorch_lightning as pl 
import torch  
import torch.nn as nn   
from torchmetrics import Accuracy  






from torch import optim
from torchvision import transforms
import numpy as np

from sklearn.decomposition import PCA


from torch.utils.data import random_split, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from pytorch_lightning.loggers import TensorBoardLogger



def GetDataModel(inp_s,outp_s):

    class DataModuleClass(pl.LightningDataModule):
        def __init__(self,lims):
            super().__init__()
            self.batch_size = 1
            self.lims = lims
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
            self.inp_s= torch.from_numpy(np.array(inp_s,dtype='float'))
            self.outp_s= torch.from_numpy(np.array(outp_s,dtype='float'))

        def prepare_data(self):



            X = torch.as_tensor(inp_s, dtype=torch.float32)
            self.X= X
            self.Y = outp_s


            # Converting numpy array to Tensor
            self.x_train_tensor = torch.from_numpy(np.array(X,dtype='float')).float().to(device)
            self.y_train_tensor = torch.from_numpy(np.array(outp_s,dtype='float')).float().to(device)

            training_dataset = TensorDataset(self.x_train_tensor, self.y_train_tensor)

            self.training_dataset = training_dataset

        def setup(self):
            data = self.training_dataset
            self.train_data=self.training_datset

        def train_dataloader(self):
            return DataLoader(self.train_data)

        def val_dataloader(self):
            return DataLoader(self.val_data)

        def __len__(self):
            return (self.lims[1]-self.lims[0])

        def __getitem__(self,index):
            return self.inp_s[self.lims[0]+index],self.outp_s[self.lims[0]+index]
        # def test_dataloader(self):
    return DataModuleClass
    


def GetNNModel(num_input,num_output,n_layers,n_width,lr=4e-7,noise=0):
    
    model = []
    model.append(nn.Linear(num_input,n_width*n_width))
    for i in range(n_layers):
        model.append(nn.GELU())
        model.append(nn.Linear(n_width*n_width,n_width*n_width))
    model.append(nn.GELU())
    model.append(nn.Linear(n_width*n_width,num_output))
    model = nn.Sequential(*model).to(torch.float64)
    
    
    

    class PTModule(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.SimplePT=model
            #self.SimplePT2=SimplePT21

        def training_step(self, batch, batch_idx):
            # training_step defines the train loop.
            # it is independent of forward
            

            x, y = batch
            
            x_hat = self.SimplePT(x)
            #x_hat2 = self.SimplePT2(x)
            #loss = torch.abs(((y.float()-x_hat)@inv_cov.float()@(y.float()-x_hat).T).float())/10000000
            #loss = torch.abs(nn.functional.gaussian_nll_loss(x_hat,y,cov_d))
            loss = torch.abs(nn.functional.l1_loss(x_hat,y+noise*torch.randn(y.size()).to('cuda')))
            #loss = torch.abs(nn.functional.mse_loss(x_hat,y))
            # Logging to TensorBoard by default
            self.log("train_loss", loss)
            return loss
        def validation_step(self, batch, batch_idx):
            # training_step defines the train loop.
            # it is independent of forward
            
            x, y = batch


            x_hat = self.SimplePT(x)
            #x_hat2 = self.SimplePT2(x)
            #loss = torch.abs(((y.float()-x_hat)@inv_cov.float()@(y.float()-x_hat).T).float())/10000000
            #loss = torch.abs(nn.functional.gaussian_nll_loss(x_hat,y,cov_d))
            loss = torch.abs(nn.functional.l1_loss(x_hat,y))
            #loss = torch.abs(nn.functional.mse_loss(x_hat,y))
            # Logging to TensorBoard by default
            self.log("val_loss", loss)

        def configure_optimizers(self):
            optimizer = optim.Adam(self.parameters(), lr=lr)
            return optimizer
        def training_epoch_end(self, outputs) -> None:
            loss = sum(output['loss'] for output in outputs) / len(outputs)
            #logging.warning(loss)
        def forward(self,x):
            x_hat = self.SimplePT(x)
            #x_hat2 = self.SimplePT2(x)
            return x_hat
    return PTModule


    



    
    




class NNParameterModelPCA:
    def __init__(self, idt_s,dt_s,n_layers,n_width, pca_components = 'mle',svd_solver='full',lr=4e-7,noise = 0):
        self.idt_s = idt_s
        self.dt_s = dt_s
        
        
        
        self.scaler = StandardScaler().fit((np.array(self.idt_s).T).T)

        self.scaler2 = StandardScaler().fit((np.array(self.dt_s).T).T)
        
        
        self.pca = PCA(n_components=pca_components,svd_solver=svd_solver)
        self.pca.fit(self.scaler2.transform(self.dt_s))
        
        self.outp_s_conv = torch.as_tensor(self.pca.transform(self.scaler2.transform((np.array(self.dt_s).T).T)), dtype=torch.float32)

        
        
        self.inp_s_conv = torch.as_tensor(self.scaler.transform((np.array(self.idt_s).T).T), dtype=torch.float32)
        num_input = np.shape(self.inp_s_conv)[1]
        
        num_output = np.shape(self.outp_s_conv)[1]
        self.model = GetNNModel(num_input,num_output,n_layers,n_width,lr=lr,noise=noise)()
        
    def Optimise(self,logs = "/pscratch/sd/s/strusov/NN_logs/ez_NN",epochs=5000,training=3000,validation=200,batch_size = 25,autograd=1):
    

        batch_size=batch_size
        num_workers=1

        dt_loader = GetDataModel(self.inp_s_conv,self.outp_s_conv)([0,training])

        dt_loader_val = GetDataModel(self.inp_s_conv,self.outp_s_conv)([training,training+validation])

        dataset = DataLoader(dt_loader,batch_size=batch_size, num_workers=num_workers, shuffle=True,drop_last=True)
        dataset_val = DataLoader(dt_loader_val,batch_size=22, num_workers=num_workers, shuffle=False,drop_last=False)
        logger = TensorBoardLogger(logs)
        trainer = pl.Trainer(max_epochs=epochs, accelerator="gpu",accumulate_grad_batches=autograd,auto_lr_find=True,devices=1,logger=logger)

        trainer.fit(self.model,dataset,dataset_val)
        
        
    def Predict(self,inp_dt):
        
        
        if(len(np.shape(inp_dt))==1):
            inp_dt = [inp_dt]
        tst_out = self.scaler2.inverse_transform(self.pca.inverse_transform(self.model.to('cuda')(torch.tensor(self.scaler.transform(inp_dt),dtype=float).to('cuda')).cpu().detach().numpy()))
        
        
        return tst_out
        
    
    
        

    