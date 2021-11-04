import torch
from torch.utils import data
import numpy as np
# from netCDF4 import Dataset
import copy
import time
# import os
from dataset.utils.transforms import *
import matplotlib.pyplot as plt
from matplotlib.pylab import datestr2num


class WData(data.Dataset):
    def __init__(self, data_list=None, transforms=None, input_length=1, output_length=0, interval=3):
        self.group(data_list)
        self.transforms = transforms
        self.input_length = input_length
        self.output_length = output_length
        self.time_table = self.read_txt()
        self.interval = interval

    def __len__(self):
        return len(self.value_list)

    def __getitem__(self, idx):
        data_every_moment = {}
        start_time = self.value_list[idx].split('/')[-1].split('_')[-2]
        time_length = self.input_length+self.output_length
        sequence_date = self.generate_sequence(start_time, time_length, self.interval)
        file_list = self.generate_file_root(sequence_date, self.value_list[idx])

        for i in range(1, time_length+1):
            data_every_moment[sequence_date[i * 3 - 1]] = 0
            for j in range(1, self.interval+1):
                data_every_moment[sequence_date[i * self.interval - 1]] += np.load((file_list[i*self.interval-j]))[0]
        data_every_moment = self.transforms(*data_every_moment.values())

        return [self.classify3h(temp) for temp in data_every_moment[0]], sequence_date[-1]
                # data_every_moment[0:self.input_length],\

    def read_txt(self):
        temp = []
        with open('/home/jinqizhao2/PycharmProjects/Sequence_Precipitation/dataset/LAPS/time_table.txt', 'r') as f:
            for l in f:
                temp.append(l.split('\n')[0])
        f.close()
        order = [i for i in range(len(temp))]
        dic = dict(zip(temp, order))
        return dic

    def generate_sequence(self, start_date, length, interval):
        l = [start_date]
        start_number = self.time_table[start_date]
        for i in range(1, length*interval):
            l.append(list(self.time_table.keys())[start_number+i])
        return l

    def classify3h(self, r):
        if isinstance(r, torch.Tensor):
            r1 = r.numpy().copy()
        elif isinstance(r, np.ndarray):
            r1 = r.copy()
        r0 = np.zeros_like(r)
        r0[r1 <= 0.1] = 0
        r0[np.logical_and(r1 > 0.1, r1 <= 3)] = 1
        r0[np.logical_and(r1 > 3, r1 <= 10)] = 2
        r0[np.logical_and(r1 > 10, r1 <= 20)] = 3
        r0[np.logical_and(r1 > 20, r1 <= 9990)] = 4
        r0[r1 > 9990] = -1
        r0[np.isnan(r1)] = -1

        return torch.from_numpy(r0).long()

    def group(self, data_list):
        self.value_list = []
        for i in data_list:
            if "AIWSRPF_" in i:
                self.value_list.append(i)

    def generate_file_root(self, sequence_date, start_file_root):
        temp = []
        for i in sequence_date:
            s = start_file_root.split('/')
            s[5] = i[0:4]
            s[6] = i[0:8]
            s[7] = s[7].replace(sequence_date[0], i)
            temp.append('/'.join(s))
        return temp


if __name__ == '__main__':
    transform_train = Compose([
        # RandomCrop(320),
        ToTensor(),
    ])
    # label_files = []
    #
    # label_root = "/home/zhangxinbang/datasets/weather_dataset_npy/"
    # for root, dir, files in os.walk(label_root):
    #     for file in files:
    #         label_files.append(os.path.join(root, file))

    label_root = "/home/zhangxinbang/datasets/weather_dataset_save/"

    # label_files = effective_sample(label_root, "202002010000", 180*24)
    label_files = []
    with open('LAPS/effective_label.txt') as f:
        for i in f:
            label_files.append(i.split('\n')[0])

    dataset = WData(label_files, transform_train)
    dataloader = data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)

    start = time.time()
    record = {}
    for i, (input, date) in enumerate(dataloader):
        # input = [temp.cuda() for temp in input]
        # input_next = [temp.cuda() for temp in input_next]
        for j, l in zip(input[0], date):
            statics = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}
            for k in statics.keys():
                statics[k] += (j == k).sum()
            record[l] = list(statics.values())
        print(i)

    print(time.time() - start)

    date_sum_dic = {}
    for i in record.keys():
        print(i)
        if i[0:6] not in list(date_sum_dic.keys()):
            date_sum_dic[i[0:8]] = [record[i][0], record[i][1], record[i][2], record[i][3], record[i][4]]
        else:
            date_sum_dic[i[0:8]][0] += record[i][0]
            date_sum_dic[i[0:8]][1] += record[i][1]
            date_sum_dic[i[0:8]][2] += record[i][2]
            date_sum_dic[i[0:8]][3] += record[i][3]
            date_sum_dic[i[0:8]][4] += record[i][4]

    for k in date_sum_dic.keys():
        print(k)
        print(date_sum_dic[k][0]/date_sum_dic[k][-1])

    x_list = list(date_sum_dic.keys())[80:100]
    # x_list = [datestr2num(i) for i in x_list]
    x_ = range(len(x_list))
    y_list = []
    for k in date_sum_dic.keys():
        y_list.append(date_sum_dic[k][0]/date_sum_dic[k][-1])
    y_list = y_list[80:100]

    plt.plot_date(x_list, y_list, 'ro-')
    plt.show()

    # print(date_sum_dic)
    # print('finish')

    # for k in statics.keys():
    #     print(statics[k])

