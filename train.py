import os
import os.path as path
import argparse
from Preprocess.dataset import get_dataset
from Preprocess.utils import get_config,update_config,save_config,read_dir,add_post
from torch.utils.data import DataLoader
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
import matplotlib.pyplot as plt
from model.model import UNet

class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)
if __name__ == "__main__":

    ## 确定配置文件
    parser = argparse.ArgumentParser(description="Train the Net")
    parser.add_argument("--default_config",default="config/adn.yaml",help="default configs")
    parser.add_argument("--run_config",default="runs/adn.yaml",help="run configs")
    args = parser.parse_args()                                                                                    

    #Get  options
    opts = get_config(args.default_config)
    run_opts = get_config(args.run_config)
    run_opts = run_opts["deep_lesion"]["train"]
    update_config (opts,run_opts)
    run_dir = path.join(opts["checkpoints_dir"],"deep_lesion")
    if not path.isdir(run_dir): os.makedirs(run_dir)
    save_config(opts,path.join(run_dir,"train_options.yaml"))
    def get_image(data):
        dataset_type = dataset_opts['dataset_type']
        if dataset_type == "deep_lesion":
            if dataset_opts[dataset_type]['load_mask']: return data['lq_image'], data['hq_image'], data['mask']
            else: return data['lq_image'], data['hq_image']
        elif dataset_type == "spineweb":
            return data['a'], data['b']
        elif dataset_type == "nature_image":
            return data["artifact"], data["no_artifact"]
        else:
            raise ValueError("Invalid dataset type!")
    #Get dataset
    dataset_opts = opts['dataset']
    train_dataset = get_dataset(**dataset_opts)
    train_loader = DataLoader(train_dataset,
        batch_size=opts["batch_size"], num_workers=0, shuffle=True)#num_workrt debug=0 run =2
    criterion = sum_squared_error()
    
    for data in train_loader:
        mask = data['mask'][0].resize(256,256)
        image = data['lq_image'][0].resize(256,256)
        mask= 1-mask
        model = UNet()
        output = model(mask*image)
        print(output.shape)







