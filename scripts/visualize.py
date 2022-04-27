import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import numpy as np

def show(grids, name):

    fig, axs = plt.subplots(nrows=len(grids), squeeze=False)
    # fig.set_size_inches(25,8)

    for i, grid in enumerate(grids):
        grid = grid.detach()
        grid = F.to_pil_image(grid)
        
        axs[0, i].imshow(np.asarray(grid))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.savefig(f"{name}.png", format="png", bbox_inches="tight")
#     fix.show()

def visualize_results(model, test_loader, device):
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
        grid_gt = make_grid(full_gt_seq[idx], 5)
        grid_out = make_grid(predictions[idx], 5)
        visual_grid.append(grid_gt, grid_out)

    show(visual_grid, "grid")

    