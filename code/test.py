import argparse
import logging
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from networks.unet_model import UNet
from networks.kan_unet import KAN_UNet
from dataloaders.dataloader import BreastSegmentation
import dataloaders.custom_transforms as tr
from utils.AverageMeter import RunningMetrics
from medpy.metric import binary
from utils import metrics

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
args = parser.parse_args()

part = ['base'] 
dataset = BreastSegmentation
n_part = len(part)
dice_calcu = {'breast':metrics.dice_coeff}
running_metrics =  RunningMetrics(2)

@torch.no_grad()
def test(args, model, test_dataloader, epoch):
    model.eval()
    val_dice = [0.0] * n_part
    val_dc, val_jc, val_hd, val_asd = [0.0] * n_part, [0.0] * n_part, [0.0] * n_part, [0.0] * n_part
    val_F1 = [0.0] * n_part
    domain_num = len(test_dataloader)
    num = 0
    for i in range(domain_num):
        cur_dataloader = test_dataloader[i]
        domain_val_dice = [0.0] * n_part
        domain_val_dc, domain_val_jc, domain_val_hd, domain_val_asd = [0.0] * n_part, [0.0] * n_part, [0.0] * n_part, [0.0] * n_part
        domain_code = i+1
        c_matrix = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
        for batch_num, sample in enumerate(cur_dataloader):
            data = sample['image'].cuda()
            mask = sample['label'].cuda()
            mask = mask.eq(0).long()
            output = model(data)

            _, cd_preds = torch.max(output, 1)
            running_metrics.update(mask.data.cpu().numpy(), cd_preds.data.cpu().numpy())
            labels_np = mask.data.cpu().numpy()
            preds_np = cd_preds.data.cpu().numpy()
            tp = np.sum((labels_np == 0) & (preds_np == 0))
            tn = np.sum((labels_np == 1) & (preds_np == 1))
            fp = np.sum((labels_np == 1) & (preds_np == 0))
            fn = np.sum((labels_np == 0) & (preds_np == 1))
            c_matrix['tn'] += tn
            c_matrix['fp'] += fp
            c_matrix['fn'] += fn
            c_matrix['tp'] += tp
            
            mask = mask.cpu()
            output = output.cpu()
            pred_label = torch.max(torch.softmax(output, dim=1), dim=1)[1]
            pred_onehot = pred_label.clone().unsqueeze(1)
            mask_onehot = mask.clone().unsqueeze(1)
                
            dice = dice_calcu[args.dataset](np.asarray(pred_label),mask)
            
            dc, jc, hd, asd = [0.0] * n_part, [0.0] * n_part, [0.0] * n_part, [0.0] * n_part
            for j in range(len(data)):
                for i, p in enumerate(part):
                    dc[i] += binary.dc(np.asarray(pred_onehot[j,i], dtype=bool),
                                            np.asarray(mask_onehot[j,i], dtype=bool))
                    jc[i] += binary.jc(np.asarray(pred_onehot[j,i], dtype=bool),
                                            np.asarray(mask_onehot[j,i], dtype=bool))
                    if pred_onehot[j,i].float().sum() < 1e-4:
                        hd[i] += 100
                        asd[i] += 100
                    else:
                        hd[i] += binary.hd95(np.asarray(pred_onehot[j,i], dtype=bool),
                                            np.asarray(mask_onehot[j,i], dtype=bool))
                        asd[i] += binary.asd(np.asarray(pred_onehot[j,i], dtype=bool),
                                            np.asarray(mask_onehot[j,i], dtype=bool))
            for i, p in enumerate(part):
                dc[i] /= len(data)
                jc[i] /= len(data)
                hd[i] /= len(data)
                asd[i] /= len(data)
            for i in range(len(domain_val_dice)):
                domain_val_dice[i] += dice[i]
                domain_val_dc[i] += dc[i]
                domain_val_jc[i] += jc[i]
                domain_val_hd[i] += hd[i]
                domain_val_asd[i] += asd[i]
        
        score = running_metrics.get_scores()
        tn, fp, fn, tp = c_matrix['tn'], c_matrix['fp'], c_matrix['fn'], c_matrix['tp']
        P = tp / (tp + fp)
        R = tp / (tp + fn)
        F1 = 2 * P * R / (R + P)
        IOU0 = tn/(tn+fp+fn)
        IOU1 = tp/(tp+fp+fn)
        mIOU = (IOU0+IOU1)/2
        OA = (tp+tn)/(tp+fp+tn+fn)
        p0 = OA
        pe = ((tp+fp)*(tp+fn)+(fp+tn)*(fn+tn))/(tp+fp+tn+fn)**2
        Kappa = (p0-pe)/(1-pe)
        print('Domain: {}\nPrecision: {}\nRecall: {}\nF1-Score: {} \nIOU0:{} \nIOU1:{} \nmIOU:{}'.format(domain_code, P, R, F1, IOU0, IOU1, mIOU))
        print('OA: {}\nKappa: {}'.format(OA,Kappa))
        val_F1 += F1    
        for i in range(len(domain_val_dice)):
            domain_val_dice[i] /= len(cur_dataloader)
            val_dice[i] += domain_val_dice[i]
            domain_val_dc[i] /= len(cur_dataloader)
            val_dc[i] += domain_val_dc[i]
            domain_val_jc[i] /= len(cur_dataloader)
            val_jc[i] += domain_val_jc[i]
            domain_val_hd[i] /= len(cur_dataloader)
            val_hd[i] += domain_val_hd[i]
            domain_val_asd[i] /= len(cur_dataloader)
            val_asd[i] += domain_val_asd[i]
        text = 'domain%d epoch %d :' % (domain_code, epoch)
        text += '\n\t'
        for n, p in enumerate(part):
            text += 'val_%s_dice: %f, ' % (p, domain_val_dice[n])
        text += '\n\t'
        for n, p in enumerate(part):
            text += 'val_%s_dc: %f, ' % (p, domain_val_dc[n])
        text += '\t'
        for n, p in enumerate(part):
            text += 'val_%s_jc: %f, ' % (p, domain_val_jc[n])
        text += '\n\t'
        for n, p in enumerate(part):
            text += 'val_%s_hd: %f, ' % (p, domain_val_hd[n])
        text += '\t'
        for n, p in enumerate(part):
            text += 'val_%s_asd: %f, ' % (p, domain_val_asd[n])
        logging.info(text)
        
    model.train()
    for i in range(len(val_dice)):
        val_dice[i] /= domain_num
        val_dc[i] /= domain_num
        val_jc[i] /= domain_num
        val_hd[i] /= domain_num
        val_asd[i] /= domain_num
    text = 'Average Score:'
    text += '\n'
    text += 'val_%s_F1: %f, ' % (p, val_F1/domain_num)
    text += '\n'
    for n, p in enumerate(part):
        text += 'val_%s_dc: %f, ' % (p, val_dc[n])
    text += '\n'
    for n, p in enumerate(part):
        text += 'val_%s_jc: %f, ' % (p, val_jc[n])
    text += '\n'
    for n, p in enumerate(part):
        text += 'val_%s_hd: %f, ' % (p, val_hd[n])
    text += '\n'
    for n, p in enumerate(part):
        text += 'val_%s_asd: %f, ' % (p, val_asd[n])
    # logging.info(text)
    print(text)
    return val_dice, val_dc, val_jc, val_hd, val_asd
    
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
        if args.model == 'kan':
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
