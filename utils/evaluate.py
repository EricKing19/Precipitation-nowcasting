import numpy as np
import pandas as pd


def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) + label_pred[mask],
        minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist


def evaluate(gts, predictions, num_classes, name):
    hist = np.zeros((num_classes, num_classes))
    for lp, lt in zip(predictions, gts):
        hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
    # axis 0: gt, axis 1: prediction
    acc = np.diag(hist).sum() / hist.sum()
    # acc_cls = np.diag(hist) / hist.sum(axis=1)
    # acc_cls = np.nanmean(acc_cls)
    iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iou = np.nanmean(iou)
    # freq = hist.sum(axis=1) / hist.sum()
    # fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()

    # save hist to csv
    if num_classes == 2:
        row_name = ['label_0', 'label_1', 'pred_sum', 'precision']
        col_name = ['pred_0', 'pred_1', 'label_sum', 'recall']
    else:
        row_name = ['label_0', 'label_1', 'label_2', 'label_3', 'label_4', 'pred_sum', 'precision']
        col_name = ['pred_0', 'pred_1', 'pred_2', 'pred_3', 'pred_4', 'label_sum', 'recall']
    out = np.row_stack([hist, hist.sum(axis=0)])
    out = np.column_stack([out, out.sum(axis=1)])
    recall = np.diag(out) / out[:, -1]
    recall[-1] = 0
    recall = np.append(recall, 0)
    precision = np.diag(out) / out[-1, :]
    precision[-1] = 0
    out = np.row_stack([out, precision])
    out = np.column_stack([out, recall])
    out = pd.DataFrame(index=row_name, columns=col_name, data=out)
    out.to_csv(name)

    return acc, mean_iou
