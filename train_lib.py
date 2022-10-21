import torch
import tqdm

def magnitude_regularized_complex_spectrum_loss(X_gt, X_est):
    """
    Loss function
    :param X_gt:
    :param X_est:
    :return:
    """
    alpha = 0.5 # regularized factor
    x_est_complex = torch.complex(X_est[:, :, :, 0], X_est[:, :, :, 1])
    x_gt_complex = torch.complex(X_gt[:, :, :, 0], X_gt[:, :, :, 1])
    complex_spectrum_mse = torch.sum(torch.pow(torch.abs(x_gt_complex - x_est_complex ), 2), dim=(1, 2))
    magnitude_spectrum_mse = torch.sum(torch.pow(torch.abs(torch.abs(x_gt_complex) - torch.abs(x_est_complex)), 2),
                                       dim=(1, 2))
    loss_value = alpha*complex_spectrum_mse + (1-alpha)*magnitude_spectrum_mse
    return torch.mean(loss_value)


def train_BFM(dataloader, model, loss_fn, optimizer, device, scaler):
    num_batches = len(dataloader.dataset)
    running_loss = 0
    model.train()
    for batch, (X_mix, X_trgt, X_bm) in enumerate(tqdm(dataloader)):
        X_mix, X_trgt, X_bm = X_mix.to(device), X_trgt.to(device), X_bm.to(device)

        optimizer.zero_grad(set_to_none=True)

        # We are training BFM, which has 3 outputs (X_bfm --> est spectrogram, X_embed --> embeddings, en_list --> skip connections)
        with torch.cuda.amp.autocast():
            X_bfm, _, _ = model(X_bm, X_mix)
            loss = loss_fn(X_trgt, X_bfm)

        # Backpropagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()

    return running_loss/num_batches


def val_BFM(dataloader, model, loss_fn, optimizer, device, scheduler):
    num_batches = len(dataloader.dataset)
    model.eval()
    #tic = time.time()
    val_loss = 0
    with torch.no_grad():
        for batch, (X_mix,X_trgt,X_bm) in enumerate(tqdm(dataloader)):
            X_mix, X_trgt, X_bm = X_mix.to(device), X_trgt.to(device), X_bm.to(device)

            # We are training BFM, which has 3 outputs (X_bfm --> est spectrogram, X_embed --> embeddings, en_list --> skip connections)
            X_bfm, _, _ = model(X_bm, X_mix)
            val_loss += loss_fn(X_trgt, X_bfm).item()
    val_loss/=num_batches
    scheduler.step(val_loss)
    print(f"Val Error: Avg loss: {val_loss:>8f} \n")
    return val_loss

def train_RRM(dataloader, model_BFM,model_RRM, loss_fn,optimizer,device, scaler):
    num_batches = len(dataloader.dataset)
    running_loss=0
    model_RRM.train()
    #tic = time.time()
    for batch, (X_mix, X_trgt,X_bm) in enumerate(tqdm(dataloader)):
        X_mix, X_trgt, X_bm = X_mix.to(device), X_trgt.to(device), X_bm.to(device)

        optimizer.zero_grad(set_to_none=True)

        # We are training BFM, which has 3 outputs (X_bfm --> est spectrogram, X_embed --> embeddings, en_list --> skip connections)
        with torch.cuda.amp.autocast():
            X_bfm, X_embed, en_list = model_BFM(X_bm, X_mix)
            X_rrm = model_RRM(X_embed, X_mix, en_list)
            X_bfm_rrm = X_bfm + X_rrm
            loss = loss_fn(X_trgt, X_bfm_rrm)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    return running_loss/num_batches

def val_RRM(dataloader, model_BFM,model_RRM,  loss_fn, device, scheduler):
    num_batches = len(dataloader.dataset)
    model_RRM.eval()
    #tic = time.time()
    val_loss = 0
    with torch.no_grad():
        for batch, (X_mix,X_trgt,X_bm) in enumerate(tqdm(dataloader)):
            X_mix, X_trgt, X_bm = X_mix.to(device), X_trgt.to(device), X_bm.to(device)

            # We are training BFM, which has 3 outputs (X_bfm --> est spectrogram, X_embed --> embeddings, en_list --> skip connections)
            X_bfm, X_embed, en_list = model_BFM(X_bm, X_mix)
            X_rrm = model_RRM(X_embed, X_mix, en_list)
            X_bfm_rrm = X_bfm + X_rrm
            val_loss += loss_fn(X_trgt, X_bfm_rrm)
    val_loss/=num_batches
    scheduler.step(val_loss)
    print(f"Val Error: Avg loss: {val_loss:>8f} \n")
    return val_loss
