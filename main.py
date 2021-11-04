import os
import time
import torch
import shutil
import logging
import argparse
import numpy as np
from config import cfg
import torch.utils.data as data
from dataset.LAPS_3Km_Dataset_Multi import WData
import dataset.utils.transforms as joint_transforms
from models.Encoder_Forecaster import EF
from models.encoder import Encoder
from models.forecaster import Forecaster
from models.STIN import SFNet
from experiments.net_params import encoder_params, forecaster_params
from torch.utils.data.distributed import DistributedSampler
from apex.parallel import convert_syncbn_model
from apex.parallel import DistributedDataParallel
from apex import amp
from tqdm import tqdm
from utils.evaluate import evaluate

best_record = {'epoch': 0, 'val_loss': 0.0, 'acc': 0.0, 'miou': 0.0}

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
args = parser.parse_args()


def main():
    global args, best_record

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    device = torch.device(f'cuda:{args.local_rank}')

    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    if cfg.GLOBAL.AUGMENT:
        transform_train = joint_transforms.Compose([
            joint_transforms.Crop(cfg.GLOBAL.SIZE),
            # joint_transforms.Normalize(),
            joint_transforms.ToTensor(),
        ])
        transform_val = joint_transforms.Compose([
            joint_transforms.Crop(cfg.GLOBAL.SIZE),
            # joint_transforms.Normalize(),
            joint_transforms.ToTensor(),
        ])
    else:
        transform_train = None

    if args.local_rank == 0:
        if not os.path.exists(cfg.GLOBAL.MODEL_SAVE_DIR):
            os.makedirs(cfg.GLOBAL.MODEL_SAVE_DIR)
        if not os.path.exists(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'result')):
            os.makedirs(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'result'))

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'record.log'),
                        filemode='w')

    label_files_train = []
    with open("dataset/LAPS/effective_train.txt") as f:
        for i in f:
            label_files_train.append(i.split('\n')[0])
    f.close()

    label_files_val = []
    with open("dataset/LAPS/effective_val.txt") as f:
        for i in f:
            label_files_val.append(i.split('\n')[0])
    f.close()

    dataset_train = WData(label_files_train, transform_train,
                          input_length=cfg.LAPS.IN_LEN, output_length=cfg.LAPS.OUT_LEN, interval=cfg.LAPS.STRIDE)
    train_sampler = DistributedSampler(dataset_train)
    dataloader_train = data.DataLoader(dataset_train, batch_size=cfg.GLOBAL.BATCH_SZIE, shuffle=None,
                                       drop_last=True, sampler=train_sampler)

    dataset_val = WData(label_files_val, transform_val,
                        input_length=cfg.LAPS.IN_LEN, output_length=cfg.LAPS.OUT_LEN, interval=cfg.LAPS.STRIDE)
    val_sampler = DistributedSampler(dataset_val)
    dataloader_val = data.DataLoader(dataset_val, batch_size=cfg.GLOBAL.BATCH_SZIE, shuffle=None,
                                     drop_last=True, sampler=val_sampler)

    if cfg.MODEL.NAME == 'ConvGRU':
        encoder = Encoder(encoder_params[0], encoder_params[1]).to(device)
        forecaster = Forecaster(forecaster_params[0], forecaster_params[1]).to(device)
        model = EF(encoder, forecaster)
    if cfg.MODEL.NAME == 'STIN':
        model = SFNet(input_channels=5, dilations=[2, 4], num_class=cfg.LAPS.NUM_CLASS, high_channels=2048,
                      low_channels=512, input_length=cfg.LAPS.IN_LEN, output_length=cfg.LAPS.OUT_LEN)

    # load pretrained
    # model_dict = model.state_dict()
    # pretrained_dict = torch.load('experiments/TrajGRU/encoder_forecaster_45000.pth')
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)

    model = convert_syncbn_model(model).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.GLOBAL.LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)

    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    model = DistributedDataParallel(model, delay_allreduce=True)

    print('Number of models parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])
    ))

    for epoch in range(cfg.GLOBAL.EPOCHS):
        scheduler.step()

        # train for one epoch
        train(dataloader_train, model, optimizer, epoch, device)

        # evaluate on validation set
        acc, mean_iou = validate(dataloader_val, model, epoch, device)
        acc, mean_iou = torch.tensor(acc).to(device), torch.tensor(mean_iou).to(device)
        torch.distributed.all_reduce(acc, op=torch.distributed.ReduceOp.SUM, async_op=True)
        torch.distributed.all_reduce(mean_iou, op=torch.distributed.ReduceOp.SUM, async_op=True)
        if mean_iou > 1:
            mean_iou /= torch.distributed.get_world_size()
        if acc > 1:
            acc /= torch.distributed.get_world_size()

        if args.local_rank == 0:
            is_best = mean_iou.cpu().numpy() > best_record['miou']
            if is_best:
                best_record['epoch'] = epoch
                best_record['acc'] = acc.cpu().numpy()
                best_record['miou'] = mean_iou.cpu().numpy()
            save_checkpoint({
                'epoch': epoch + 1,
                'accuracy': acc.cpu().numpy(),
                'miou': mean_iou.cpu().numpy(),
                'models': model.state_dict(),
            }, is_best)

            logging.info(
                '---------------------------------------------------------------------------------------------------')
            logging.info('[epoch: %d], [acc: %.5f], [miou: %.5f]' % (epoch, acc.cpu().numpy(), mean_iou.cpu().numpy()))
            logging.info('best record: [epoch: {epoch}], [acc: {acc:.5f}], [miou: {miou:.5f}]'.format(**best_record))
            logging.info(
                '---------------------------------------------------------------------------------------------------')
            print('---------------------------------------------------------------------------------------------------')
            print('[epoch: %d], [acc: %.5f], [miou: %.5f]' % (epoch, acc.cpu().numpy(), mean_iou.cpu().numpy()))
            print('best record: [epoch: {epoch}], [acc: {acc:.5f}], [miou: {miou:.5f}]'.format(**best_record))
            print('---------------------------------------------------------------------------------------------------')

        # model.module.encoder.stage2.dyconv2_leaky.update_temperature()
        # model.module.encoder.stage3.dyconv3_leaky.update_temperature()


def train(dataloader_train, model, optimizer, epoch, device, print_freq=20):
    losses = AverageMeter()
    batch_time = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(dataloader_train):
        criterion = make_criterion(epoch, device, target)
        input = [temp.to(device) for temp in input]
        target = [temp.long().to(device) for temp in target]

        # compute output
        output = model(input)
        loss = 0
        for p, t in zip(output, target):
            loss += criterion(p, t)

        # record loss
        losses.update(loss.data, input[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=50.0)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, i, len(dataloader_train),
                batch_time=batch_time, loss=losses
            ))


def validate(dataloader_val, model, epoch, device, print_freq=20):
    batch_time = AverageMeter()
    target_list = {}
    pred_list = {}
    for j in range(cfg.LAPS.OUT_LEN):
        target_list[j] = []
        pred_list[j] = []

    model.eval()

    end = time.time()

    for input, target in tqdm(dataloader_val):
        with torch.no_grad():
            input = [temp.to(device) for temp in input]
            target = [temp.to(device) for temp in target]

        # compute output
        # output0, output1, output2, output3 = models(input_var)
        output = model(input)

        # measure accuracy and record loss
        batch_time.update(time.time() - end)
        end = time.time()
        for m in range(len(target)):
            for n in range(target[m].shape[0]):
                target_list[m].append(target[m].cpu()[n].numpy())
                pred_list[m].append(np.argmax(output[m].cpu()[n].detach().numpy(), axis=0))

    acc = {}
    mean_iou = {}
    for j in range(cfg.LAPS.OUT_LEN):
        acc[j], mean_iou[j] = evaluate(target_list[j], pred_list[j], cfg.LAPS.NUM_CLASS,
                                       '{}/result/{}_{}.csv'.format(cfg.GLOBAL.MODEL_SAVE_DIR, epoch, j))

    return np.mean(list(acc.values())), np.mean(list(mean_iou.values()))


class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def make_criterion(epoch, device, target=None, date=None):
    if cfg.GLOBAL.TRAIN_RULE == 'None':
        per_cls_weights = None
    elif cfg.GLOBAL.TRAIN_RULE == 'Reweight':
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, cfg.LAPS.CLS_NUM)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cfg.LAPS.CLS_NUM)
        per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
    elif cfg.GLOBAL.TRAIN_RULE == 'BatchReweight':
        statics = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        for i in target:
            for j in statics.keys():
                statics[j] += (i == j).sum()
        cls_num = list(statics.values())
        # if cls_num[0] / cls_num[-1] > 1000:
        #     cls_num = [cfg.LAPS.IN_LEN*cfg.GLOBAL.BATCH_SZIE*cfg.GLOBAL.SIZE*cfg.GLOBAL.SIZE*i for i in cfg.LAPS.CLS_RATIO]
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, cls_num)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num)
        per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
    elif cfg.GLOBAL.TRAIN_RULE == 'MonthlyReweight':
        cls_num = [torch.from_numpy(np.load('./dataset/LAPS/' + i + '.npy')) + 1 for i in date]
        cls_num = torch.stack(cls_num, 0)
        beta = 0.9999
        effective_num = 1.0 - torch.pow(beta, cls_num)
        per_cls_weights = (1.0 - beta) / effective_num
        per_cls_weights = per_cls_weights / torch.unsqueeze(torch.sum(per_cls_weights, 1), 1) * len(cfg.LAPS.CLS_NUM)
        per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
    elif cfg.GLOBAL.TRAIN_RULE == 'DRW':
        idx = epoch // 12
        betas = [0, 0.9999]
        effective_num = 1.0 - np.power(betas[idx], cfg.LAPS.CLS_NUM)
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cfg.LAPS.CLS_NUM)
        per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)

    if cfg.GLOBAL.LOSS == 'CE' and cfg.GLOBAL.TRAIN_RULE != 'MonthlyReweight':
        criterion = torch.nn.CrossEntropyLoss(weight=per_cls_weights, ignore_index=-1).to(device)

    return criterion


def save_checkpoint(model, is_best, name=cfg.GLOBAL.MODEL_SAVE_DIR, filename='checkpoing.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "%s/" % (name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(model, filename)
    if is_best:
        shutil.copyfile(filename, '%s/' % name + 'model_best.pth.tar')


if __name__ == '__main__':
    main()
