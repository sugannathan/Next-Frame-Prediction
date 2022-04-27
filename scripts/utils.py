import os
import torch 
import numpy as np
from tqdm import tqdm
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import wandb
import yaml
import torchvision.transforms.functional as F
from piqa import SSIM
from ignite.handlers.param_scheduler import ConcatScheduler

def train_epoch(model, train_loader, optimizer, criterion, epoch, device, add_ssim):
    """ Training a model for one epoch """
    
    loss_list = []
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for idx, (seq, target) in progress_bar:
        
        seq = seq.type(torch.FloatTensor).to(device)
        target = target.type(torch.FloatTensor).to(device)
        
        optimizer.zero_grad()
        
        predictions = model(seq)
        predictions = predictions.to(device)

        full_seq = torch.cat((seq, target), dim=1)
        
        if add_ssim:
            ssim_loss = ssim_eval(predictions,full_seq,device)
            loss = criterion(predictions, full_seq) + 0.001 * ssim_loss #Lambda=0.01
        else:
            loss = criterion(predictions, full_seq)
        loss_list.append(loss.item())

        loss.backward()
         
        optimizer.step()
        
        progress_bar.set_description(f"Epoch {epoch+1} Iter {idx+1}: loss {loss.item():.5f}. ")
        if idx % 50 == 0:
            wandb.log({"loss": loss})
        
    mean_loss = np.mean(loss_list)

    return mean_loss, loss_list


@torch.no_grad()
def eval_model(model, eval_loader, criterion, device, epoch, result_path):
    """ Evaluating the model for either validation or test """
    loss_list = []
    pbar = tqdm(enumerate(eval_loader), total=len(eval_loader))
    for idx, (seq, target) in pbar:
        
        seq = seq.type(torch.FloatTensor).to(device)
        target = target.type(torch.FloatTensor).to(device)
        
        predictions = model(seq)
        predictions = predictions.to(device)
                
        loss = criterion(predictions[:, 10:, :, :, :], target)
        loss_list.append(loss.item())
            
        pbar.set_description(f"Test loss: loss {loss.item():.2f}")
        
    mean_loss = np.mean(loss_list)
    visualize_results(model, eval_loader, device, epoch, result_path)
    
    return mean_loss


def train_model(model, optimizer, scheduler, criterion, train_loader,\
                valid_loader, num_epochs, device, args, config):
    """ Training a model for a given number of epochs"""
    
    train_loss = []
    val_loss =  []
    loss_iters = []
    epochs = []
    
    torch.onnx.export(model, torch.randn(1, 10, 1, 64, 64, device="cuda"), "model.onnx", opset_version=11)
    wandb.save("model.onnx")

    for epoch in range(num_epochs):
                 
        # validation epoch
        model.eval()  # important for dropout and batch norms
        loss = eval_model(
                    model=model, eval_loader=valid_loader,
                    criterion=criterion, device=device, epoch=epoch,
                    result_path=config.result_path)
        val_loss.append(loss)

        # training epoch
        model.train()  # important for dropout and batch norms
        mean_loss, cur_loss_iters = train_epoch(
                model=model, train_loader=train_loader, optimizer=optimizer,
                criterion=criterion, epoch=epoch, device=device, add_ssim=args.add_ssim)

        train_loss.append(mean_loss)
        loss_iters = loss_iters + cur_loss_iters
  
        epochs.append(epoch+1)
        
        if isinstance(scheduler,ConcatScheduler):
          scheduler(None)
        else:
          scheduler.step(loss)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"    Train loss: {round(mean_loss, 5)}")
        print(f"    Valid loss: {round(loss, 5)}")
        print("\n")

        if (epoch+1) % config.save_freq == 0 or epoch == 0:
            saving_model(model, optimizer, epoch, config.save_path)

        wandb.log({"train_epoch_loss": mean_loss, "val_loss": loss})
    
    print(f"Training completed")
    return train_loss, val_loss, loss_iters, epochs

def ssim_eval(predictions, target, device):
    ssim = SSIM().to(device)
    pred_new=predictions.repeat(1,1,3,1,1).flatten(0,1)
    target_new=target.repeat(1,1,3,1,1).flatten(0,1)
    ssim_loss = 1 - ssim(pred_new, target_new)
    return ssim_loss


def saving_model(model, optimizer, epoch, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = f"{save_path}/model_{epoch+1}.pth"
    torch.save({
        'epoch' : epoch,
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 'stats': stats
    }, save_path)
    

def loading_model(model, path):
    optimizer = torch.optim.Adam(params=model.parameters(), lr= 3e-4)
    checkpoint =  torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    # stats = checkpoint['stats']
    return model, optimizer, epoch


def save_results(grid, name):
    fix, axs = plt.subplots()
    fix.set_size_inches(25,8)

    axs.imshow(grid.cpu().numpy().transpose(1,2,0))
    axs.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fix.savefig(f"{name}.png", format="png", bbox_inches="tight")
    wandb.log({"outputs" : wandb.Image(grid.cpu())}) 


def show(grids, name, result_path):

    fig, axs = plt.subplots(nrows=len(grids), squeeze=False)
    fig.set_size_inches(25,8)

    for i, grid in enumerate(grids):
        grid = grid.detach()
        grid = F.to_pil_image(grid)
        
        axs[i, 0].imshow(np.asarray(grid))
        axs[i, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    fig.savefig(f"{result_path}/{name}.png", format="png", bbox_inches="tight")
    wandb.log({"outputs" : wandb.Image(fig)}) 

def visualize_results(model, test_loader, device, epoch, result_path):
    test_input, test_target = next(iter(test_loader))
    
    test_input = test_input.to(device)
    test_target = test_target.to(device)
    
    full_gt_seq = torch.cat((test_input, test_target), dim=1)

    model.eval() 
    with torch.no_grad():
        predictions = model(test_input)
        predictions = predictions.to(device)
    
    visual_grid = []
    for idx in range(0, 5):
        grid_gt = make_grid(full_gt_seq[idx], 20)
        grid_out = make_grid(predictions[idx], 20)
        visual_grid.append(grid_gt)
        visual_grid.append(grid_out)
    show(visual_grid, f"grid_{epoch}", result_path)


def load_cfg(name):
    path = os.path.join("configs", name)
    with open(path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data


def load_dataset(opt):
    if opt.dataset == 'smmnist':
        from dataset.moving_mnist import MovingMNIST
        train_data = MovingMNIST(
                train=True,
                data_root=opt.dataset_path,
                seq_len=opt.n_past+opt.n_future,
                image_size=opt.image_width,
                deterministic=False,
                num_digits=opt.num_digits)
        test_data = MovingMNIST(
                train=False,
                data_root=opt.dataset_path,
                seq_len=opt.n_eval,
                image_size=opt.image_width,
                deterministic=False,
                num_digits=opt.num_digits)
    elif opt.dataset == 'kth':
        from dataset.kth import KTH 
        transform = transforms.Compose([transforms.Resize((64, 64))])
        train_data = KTH(
            directory=opt.dataset_path,
            transform=transform,
            download=False,
            train=True)

        test_data = KTH(
            directory=opt.dataset_path,
            transform=transform,
            download=False,
            train=False)

    return train_data, test_data


##Visualise functions for Notebooks
def showfornb(grid, seq, evaluation):
    fix, axs = plt.subplots()
    fix.set_size_inches(25,10)

    axs.imshow(grid.cpu().numpy().transpose(1,2,0))
    axs.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    
    if seq=='gt':
        if evaluation==False:
            axs.set_title('Input Sequence')
        else:
            axs.set_title('Target sequence')
    else:
        if evaluation==False:
            axs.set_title('Target sequence')
        else:
            axs.set_title('Predicted sequence')

def visualise_sample(sample, model, device, evaluation):    
    if evaluation==False:
        idx = torch.randint(len(sample), size=(1,)).item()
        seq, target = sample[idx]
        if(type(seq)!=torch.Tensor):
            seq=torch.from_numpy(seq)
            target=torch.from_numpy(target)
            
        seq = seq.to(device)
        target = target.to(device)
        
        grid_gt = make_grid(seq,seq.shape[0])
        
        showfornb(grid_gt, "gt",evaluation)
        
        grid_out = make_grid(target,target.shape[0])
        
        showfornb(grid_out,"output",evaluation)
        
    else:
        seq, target = sample
        
        seq = seq.to(device)
        target = target.to(device)
        
        model.eval() 
        with torch.no_grad():
            predictions = model(seq)
            predictions = predictions.to(device)
            
        grid_gt = make_grid(target[0], target.shape[0])
        showfornb(grid_gt, "gt", evaluation)
    
        grid_out = make_grid(predictions[:,10:,:,:,:][0], predictions[:,10:,:,:,:].shape[0])
        
        showfornb(grid_out, "output", evaluation)

