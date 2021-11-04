import os
import time
import torch
import numpy as np
from config import cfg
from apex import amp
from utils.evaluate import evaluate


def train(dataloader_train, model, criterion, optimizer, epoch, device, scheduler, print_freq=20):
    losses = AverageMeter()
    batch_time = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(dataloader_train):
        # t = [LabelTrans(target, k) for k in range(1, 5)]
        scheduler.step()
        input = torch.stack(input, dim=0)
        input = input.to(device)
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
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=50.0)
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


def validate(dataloader_val, model, criterion, epoch, device, print_freq=20):
    batch_time = AverageMeter()
    losses = AverageMeter()
    target_list = []
    pred_list = []

    model.eval()

    end = time.time()

    for i, (input, target) in enumerate(dataloader_val):
        with torch.no_grad():
            input = torch.stack(input, dim=0)
            input = input.to(device)
            input_var = torch.autograd.Variable(input, requires_grad=False)
            target = [temp.to(device) for temp in target]
            target_var = [torch.autograd.Variable(temp.long(), requires_grad=False) for temp in target]

        # compute output
        # output0, output1, output2, output3 = models(input_var)
        output = model(input_var)
        loss = 0
        for p, t in zip(output, target_var):
            loss += criterion(p, t)

        # measure accuracy and record loss
        losses.update(loss.data, input[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        for m in range(len(target)):
            for n in range(target[m].shape[0]):
                target_list.append(target[m].cpu()[n].numpy())
                pred_list.append(np.argmax(output[m].cpu()[n].detach().numpy(), axis=0))

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                i, len(dataloader_val), batch_time=batch_time, loss=losses))

    acc, mean_iou = evaluate(target_list, pred_list, cfg.LAPS.NUM_CLASS,
                             '{}/result/{}.csv'.format(cfg.GLOBAL.MODEL_SAVE_DIR, epoch))
    return acc, mean_iou, losses


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
