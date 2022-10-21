import torch
import network_lib
import train_lib
import data_lib
torch.cuda.empty_cache()
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

"""
Beam Filter Module
"""
torch.backends.cudnn.benchmark = True

writer = SummaryWriter()
# Get cpu or gpu for training
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def magnitude_regularized_complex_spectrum_loss(X_gt, X_est):
    """
    Loss function
    :param X_gt:
    :param X_est:
    :return:
    """
    alpha = 0.5 # regularized factor
    X_est_complex = torch.complex(X_est[:, :, :, 0], X_est[:, :, :, 1])
    X_gt_complex = torch.complex(X_gt[:, :, :, 0], X_gt[:, :, :, 1])
    complex_spectrum_mse = torch.sum(torch.pow(torch.abs(X_gt_complex - X_est_complex ), 2), dim=(1, 2))
    magnitude_spectrum_mse = torch.sum(torch.pow(torch.abs(torch.abs(X_gt_complex) - torch.abs(X_est_complex) ), 2),dim=(1,2))
    loss_value = alpha*complex_spectrum_mse + (1-alpha)*magnitude_spectrum_mse
    return torch.mean(loss_value)


# NOW LET'S START TRAINININGGGGG
"""
Create Dataloader
"""
batch_size = 32*torch.cuda.device_count()
train_loader = torch.utils.data.DataLoader(
    data_lib.train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=torch.cuda.device_count()*4
)

val_loader = torch.utils.data.DataLoader(
    data_lib.val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=torch.cuda.device_count()*4
)


import datetime
model_save_path = 'models'
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_bfm_path = os.path.join(model_save_path, "model_BFM_"+curr_time+".pth")
model_rrm_path = os.path.join(model_save_path, "model_RRM_"+curr_time+".pth")


# Optimizer
"""
TRAIN BFM
"""

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model_BFM = torch.nn.DataParallel(network_lib.BFM())  # .to(device)
  model_BFM = model_BFM.to(device)
else:
    print('PD')
    model_BFM = network_lib.BFM().to(device)

lr = 5e-4
optimizer_BFM = torch.optim.Adam(model_BFM.parameters(), lr=lr)
scheduler_BFM = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_BFM, 'min', factor=0.5, patience=2)


scaler = torch.cuda.amp.GradScaler()
loss_fn = magnitude_regularized_complex_spectrum_loss

epochs = 0
val_loss = None
val_loss_best = None
print('Training BFM')
for t in tqdm(range(epochs)):
    print(f"Epochs {t+1}\n-----------------------------")
    train_loss = train_lib.train_BFM(train_loader, model_BFM, loss_fn, optimizer_BFM, device, scaler)
    val_loss = train_lib.val_BFM(val_loader, model_BFM, loss_fn, optimizer_BFM,device, scheduler_BFM)
    writer.add_scalar('Loss/train_BFM',  train_loss, t)
    writer.add_scalar('Loss/val_BFM', val_loss, t)

    # Handle saving best mofel
    if t ==0:
        val_loss_best = val_loss
    if t >0 and val_loss < val_loss_best:
        torch.save(model_BFM.state_dict(), model_bfm_path)
        val_loss_best = val_loss
        print(f'Model saved epoch{t}')

"""
TRAIN RRM
"""

# Freeze BFM models
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs! BFM inference")
  model_BFM = torch.nn.DataParallel(network_lib.BFM())  # .to(device)
  model_BFM.load_state_dict(torch.load(model_bfm_path))
  model_BFM = model_BFM.to(device)
else:
    print('PD')
    model_BFM = network_lib.BFM()
    model_BFM.load_state_dict(torch.load(model_bfm_path))
    model_BFM = model_BFM.to(device)


# Freeze BFM parameters
for p in model_BFM.parameters():
    p.requires_grad = False
# Freeze BFM models
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs! RRM")
  model_RRM = torch.nn.DataParallel(network_lib.RRM())  # .to(device)
  model_RRM = model_RRM.to(device)
else:
    model_RRM = network_lib.RRM()
    model_RRM = model_RRM.to(device)

lr = 5e-4
optimizer_RRM = torch.optim.Adam(model_RRM.parameters(), lr=lr)
scheduler_RRM = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_RRM, 'min', factor=0.5, patience=2)

scaler = torch.cuda.amp.GradScaler()
loss_fn = magnitude_regularized_complex_spectrum_loss

# TRAINING LOOP RRM
epochs = 60
val_loss = None
val_loss_best = None
print('Training RRM')
for t in tqdm(range(epochs)):
    print(f"Epochs {t+1}\n-----------------------------")
    train_loss = train_lib.train_RRM(train_loader, model_BFM,model_RRM, loss_fn, optimizer_RRM,device, scaler)
    val_loss = train_lib.val_RRM(val_loader, model_BFM,model_RRM, loss_fn, device, scheduler_RRM)
    writer.add_scalar('Loss/train_RRM',train_loss,t)
    writer.add_scalar('Loss/val_RRM',val_loss,t)
    # Handle saving best mofel
    if t ==0:
        val_loss_best = val_loss
    if t >0 and val_loss < val_loss_best:
        torch.save(model_RRM.state_dict(), model_rrm_path)
        print(f'Model saved epoch{t}')
        val_loss_best = val_loss