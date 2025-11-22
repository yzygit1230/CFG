import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from networks.unet_model import UNet
from networks.kan_unet import KAN_UNet
from dataloaders.dataloader import BreastSegmentation
import dataloaders.custom_transforms as tr
from utils import util

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='kan')
parser.add_argument('--dataset', type=str, default='breast')
parser.add_argument("--data_dir", type=str, default="data/BreastSlice")
parser.add_argument("--pth_dir", type=str, default="breast-test/debug/kan_avg_dice_best_model.pth")
parser.add_argument("--model", type=str, default="kan")
parser.add_argument("--gpu", type=str, default='0')
parser.add_argument('--eval',type=bool, default=True)
parser.add_argument("--test_bs", type=int, default=1)
parser.add_argument('--domain_num', type=int, default=4)
parser.add_argument('--lb_domain', type=int, default=1)

parser.add_argument('--save_img', default=True, action='store_true')
args = parser.parse_args()

part = ['base'] 
dataset = BreastSegmentation
n_part = len(part)

@torch.no_grad()
def test(args, model, test_dataloader, epoch):
    model.eval()
    domain_num = len(test_dataloader)
    num = 0
    for i in range(domain_num):
        cur_dataloader = test_dataloader[i]
        domain_code = i+1
        for batch_num, sample in enumerate(cur_dataloader):
            data = sample['image'].cuda()
            mask = sample['label'].cuda()
            img_name = sample['img_name'][0]
            img_name = str(img_name)
            img_name = os.path.basename(img_name)
            mask = mask.eq(0).long()
            output = model(data)
            _, cd_preds = torch.max(output, 1)
            cd_preds = cd_preds.data.cpu().numpy()
            cd_preds = cd_preds.squeeze() * 255
            _, cd_preds = torch.max(output, 1)
            mask = mask.cpu()
            output = output.cpu()
            pred_label = torch.max(torch.softmax(output, dim=1), dim=1)[1]
            pred_onehot = pred_label.clone().unsqueeze(1)
            mask_onehot = mask.clone().unsqueeze(1)
            
            save_path = 'resedge/'
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save_path = 'resbinary/'
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            if args.eval and args.save_img:
                for j in range(len(data)):
                    num += 1
                    util.draw_contour_and_save(data[j], pred_onehot[j], mask_onehot[j], 'resedge/D{}_{}'.format(domain_code, img_name))
                    util.draw_binary_mask_and_save(data[j], pred_onehot[j], 'resbinary/D{}_{}'.format(domain_code, img_name))
    
def main(args):
    num_channels = 1
    num_classes = 2
    normal_toTensor = transforms.Compose([
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    domain_num = args.domain_num
    test_dataset = []
    test_dataloader = []
    for i in range(1, domain_num+1):
        cur_dataset = dataset(base_dir=args.data_dir, phase='test', splitid=-1, domain=[i], normal_toTensor=normal_toTensor)
        test_dataset.append(cur_dataset)
    for i in range(0, domain_num):
        cur_dataloader = DataLoader(test_dataset[i], batch_size = args.test_bs, shuffle=False, num_workers=0, pin_memory=True)
        test_dataloader.append(cur_dataloader)

    def create_model(ema=False):
        if args.model == 'unet':
            model = UNet(n_channels = num_channels, n_classes = num_classes)
        elif args.model == 'kan':
            model = KAN_UNet()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model.cuda()

    model = create_model()

    if args.eval:
        args.lb_domain = 1
        model.load_state_dict(torch.load(args.pth_dir))
        test(args, model, test_dataloader,args.lb_domain)
        exit()


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)
