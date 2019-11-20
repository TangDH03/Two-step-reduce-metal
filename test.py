import os
import os.path as path
import argparse
from Preprocess.dataset import get_dataset
from Preprocess.utils import get_config, update_config, save_config, read_dir, add_post
from torch.utils.data import DataLoader
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
import matplotlib.pyplot as plt
from model.model import UNet
import torch
import numpy as np
from model.DnCNN import DnCNN
from torch.nn.modules.loss import _Loss
import torch.optim as optim
from PIL import Image
class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """

    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(
            size_average, reduce, reduction)

    def forward(self, input, target):
        return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        #return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum')


if __name__ == "__main__":

    # 确定配置文件
    parser = argparse.ArgumentParser(description="Train the Net")
    parser.add_argument("--default_config",
                        default="config/adn.yaml", help="default configs")
    parser.add_argument(
        "--run_config", default="runs/adn.yaml", help="run configs")

    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
    args = parser.parse_args()

    # Get  options
    opts = get_config(args.default_config)
    run_opts = get_config(args.run_config)
    run_opts = run_opts["deep_lesion"]["train"]
    update_config(opts, run_opts)
    run_dir = path.join(opts["checkpoints_dir"], "deep_lesion")
    if not path.isdir(run_dir): os.makedirs(run_dir)
    save_config(opts, path.join(run_dir, "train_options.yaml"))
    def save_image(raw_image,output,index,gt):
        raw_image = raw_image.cpu().detach().numpy()
        #temp = raw_image[0].reshape((256,256))
        #name = "{}_iter_1.png".format(index)
        output = output.cpu().detach().numpy()
        gt = gt.cpu().detach().numpy()
        #img1 = Image.fromarray(raw_image[0].reshape((256,256)))
        name = "./test/{}_test_iter_1.png".format(index)
        plt.imsave(name,raw_image[0].reshape((256,256)),cmap="gray")
        #img2 = Image.fromarray(output[0].reshape((256,256)))
        name = "./test/{}_test_iter_2.png".format(index)
        plt.imsave(name,output[0].reshape((256,256)),cmap="gray")
        #img3 = Image.fromarray(raw_image[1].reshape((256,256)))
        name = "./test/{}_test_iter_3.png".format(index)
        plt.imsave(name,raw_image[1].reshape((256,256)),cmap="gray")
        #img4 = Image.fromarray(output[0].reshape((256,256)))
        name = "./test/{}_test_iter_4.png".format(index)
        plt.imsave(name,output[1].reshape((256,256)),cmap="gray")
        name = "./test/{}_test_iter_1_gt.png".format(index)
        plt.imsave(name,gt[0].reshape((256,256)),cmap="gray")
        name = "./test/{}_test_iter_3_gt.png".format(index)
        plt.imsave(name,gt[1].reshape((256,256)),cmap="gray")
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
    # Get dataset
    dataset_opts = opts['dataset']
    train_dataset = get_dataset(**dataset_opts)
    train_loader = DataLoader(train_dataset,
        batch_size=opts["batch_size"], num_workers=12, shuffle=True)  # num_workrt debug=0 run =2
    model = torch.load("model_050.pth")
    epoch =1
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = sum_squared_error()
    epoch_loss=0
    all_psnr = 0
    before_loss=0
    before_psnr=0
    for n_count, data in enumerate(train_loader):
                    with torch.no_grad():
                            optimizer.zero_grad()
                            gt = data['hq_image'].cuda()
                            mask = data['mask'].cuda()
                            image = data['lq_image'].cuda()
                            mask = 1-mask
                            gt = gt*mask
                            image = image*mask
                            output = model(image)
                            loss = criterion(model(image), gt)
                            before_loss=criterion( image,gt)
                
                            epoch_loss += loss.item()
                            optimizer.step()
                            cimage = image.cpu().detach().numpy()
                            psnr = 10*np.log10((255*255)/(loss.item()))
                            all_psnr+=psnr
                            before_psnr=10*np.log10((255*255)/(before_loss.item()))
                            if n_count%10==0:
                                print('%4d %4d loss = %2.4f psnr=%2.4f' % (epoch+1, n_count, loss.item()/2,psnr))
                                print('%4d %4d beforeloss = %2.4f beforepsnr=%2.4f' % (epoch+1, n_count, before_loss.item()/2,before_psnr))
                            if n_count%100==0:
                                save_image(image,output,epoch*1000+n_count,gt)
    print("epoch = %4d,loss = %4.4f ,psnr =%2.4f "% (epoch+1,epoch_loss/n_count,all_psnr/n_count))