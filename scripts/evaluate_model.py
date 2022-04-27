import os
import torch
import numpy as np
from model import Model
from utils import loading_model
import argparse
from dataset.test_moving_mnist import MovingMNIST_test
from dataset.kth import KTH
import torchvision.transforms as transforms
from tqdm import tqdm
import piqa
from skimage.metrics import structural_similarity 

def parse_commandline():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dataset', '-d', type=str)
    parser.add_argument('--model_path', '-mp', type=str)
    parser.add_argument('--save_path', '-s', type=str)

    args = parser.parse_args()
    # cfg = AttrDict(utils.load_cfg(args.cfg_name))
    return args

def all_metric(lpips_criterion, x1, x2):
    mse = torch.zeros((len(x1)))
    mae = torch.zeros((len(x1)))
    lpips = torch.zeros((len(x1)))
    # ssim = torch.zeros((len(x1)))

    for idx in range(len(x1)):
        mse[idx] =  torch.nn.functional.mse_loss(x1[idx], x2[idx])
        mae[idx] =  torch.nn.functional.l1_loss(x1[idx], x2[idx])
        lpips[idx] =  lpips_criterion(x1[idx].repeat(1, 3, 1, 1), x2[idx].repeat(1, 3, 1, 1))

    return mse, mae, lpips


def ssim_metric(x1, x2):
    ssim = torch.zeros((len(x1)))
    for idx in range(len(x1)):
        ssim[idx] =  structural_similarity(np.squeeze(x1[idx]), np.squeeze(x2[idx]))
    return ssim

def evaluate_metrices(prediction, target):

    psnr_seq = piqa.psnr.psnr(prediction[0], target[0])
    lpips_criterion = piqa.lpips.LPIPS().cuda()
    mse_seq, mae_seq, lpips_seq = all_metric(lpips_criterion, prediction[0], target[0])
    ssim_seq = ssim_metric(prediction[0].cpu().numpy(), target[0].cpu().numpy())

    return mae_seq, mse_seq, psnr_seq, lpips_seq, ssim_seq

def save_tensors(stats, save_dir ):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    for matrix in stats:
        torch.save(stats[matrix], f'{os.path.join(save_dir, matrix)}.pt')

def print_results(stats):
    print("*" * 50)
    
    for matrix in stats:
        print(f"Overall {matrix}")
        print(torch.mean(stats[matrix]).item())
        print(torch.mean(stats[matrix], dim=0))
        print("*" * 50)


def eval_model_main():
    args = parse_commandline()
    device = "cuda"
    model_path = args.model_path
    model = Model()
    model, _, _ = loading_model(model, model_path)
    model = model.to(device)


    if args.test_dataset == "moving_mnist":
        transform = transforms.Compose([transforms.ToTensor(),
                    transforms.Resize((64, 64)),
                ])

        test_set = MovingMNIST_test(root='data/mnist', train=False, download=True, transform=transform, target_transform=transform, split=10_000)
    
    if args.test_dataset == "kth":
        transform = transforms.Compose([transforms.Resize((64, 64))])
        test_set = KTH("data/kth", transform=transform, download=False, train=False)
    print(len(test_set))

    mae = torch.zeros((len(test_set), 10))
    mse = torch.zeros((len(test_set), 10))
    psnr = torch.zeros((len(test_set), 10))
    lpips = torch.zeros((len(test_set), 10))
    ssim = torch.zeros((len(test_set), 10))

    with torch.no_grad():
        
        pbar = tqdm(enumerate(test_set), total=len(test_set))
        for b_inx, (seq, target) in pbar:
            seq, target = seq.unsqueeze(0), target.unsqueeze(0)
            seq = seq.type(torch.FloatTensor).to(device)
            target = target.type(torch.FloatTensor).to(device)
            
            predictions = model(seq)
            predictions = predictions.to(device)
                    
            mae_seq, mse_seq, psnr_seq, lpips_seq, ssim_seq  = evaluate_metrices(predictions[:, 10:, :, :, :], target)
            mae[b_inx, :] = mae_seq
            mse[b_inx, :] = mse_seq
            psnr[b_inx, :] = psnr_seq
            lpips[b_inx, :] = lpips_seq
            ssim[b_inx, :] = ssim_seq

        stats = {"MAE": mae, "MSE": mse,"PSNR": psnr,"LPIPS": lpips,"SSIM": ssim}
        save_tensors(stats, args.save_path)
        print_results(stats)


if __name__ == "__main__":
    eval_model_main()