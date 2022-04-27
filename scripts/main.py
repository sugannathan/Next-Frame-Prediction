import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from lr_warmup import ReduceLROnPlateauWithWarmup
from torch.optim.lr_scheduler import ReduceLROnPlateau


from model import Model
import utils
import wandb
import argparse
from attrdict import AttrDict
from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup

def parse_commandline():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_name', '-c', type=str)
    parser.add_argument('--add_ssim', type=bool)
    parser.add_argument('--lr_warmup', type=bool)
    parser.add_argument('--criterion', type=str)
    parser.add_argument('--scheduler','-s', type=str)
    
    args = parser.parse_args()
    cfg = AttrDict(utils.load_cfg(args.cfg_name))
    return args, cfg

if __name__ == "__main__":

    args, cfg = parse_commandline()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set, test_set = utils.load_dataset(cfg)

    batch_size = cfg.batch_size

    train_loader = DataLoader(
                    dataset=train_set,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=4)

    test_loader = DataLoader(
                    dataset=test_set,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=4)
    print("Data loaders ready")

# W and B for logging grads
    wandb.init()

    model = Model()
    mdoel = model.to(device)
    wandb.watch(model, log_freq=cfg.log_freq)

    print("Model Loaded")

    if args.criterion == "mse":
        criterion = nn.MSELoss()
    elif args.criterion == "mae":
        criterion = nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr= cfg.learning_rate)
    criterion = criterion.to(device)
    if args.scheduler == "plateau":

      if args.lr_warmup:
        scheduler = ReduceLROnPlateauWithWarmup(optimizer, cfg.warmup_init_lr, cfg.learning_rate, cfg.warmup_epochs)
      else:
        scheduler = ReduceLROnPlateau(optimizer, factor=cfg.factor, patience=cfg.patience)

    elif args.scheduler == "exponential":
      
      if args.lr_warmup:
        exp_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)
        scheduler = create_lr_scheduler_with_warmup(exp_scheduler,
                                            warmup_start_value=cfg.warmup_init_lr,
                                            warmup_end_value=cfg.learning_rate,
                                            warmup_duration=cfg.warmup_epochs)
      else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)


    train_loss, test_loss, loss_iter, epochs = utils.train_model(model, optimizer, scheduler, criterion,\
                                                                train_loader, test_loader, cfg.epochs, device, args, cfg)

