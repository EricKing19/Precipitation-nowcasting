from torch.utils import data
# from netCDF4 import Dataset
import time
# import os
from dataset.utils.transforms import *


class WData(data.Dataset):
    def __init__(self, data_list=None, transforms=None, input_length=3, output_length=3, interval=3):
        self.group(data_list)
        self.transforms = transforms
        self.input_length = input_length
        self.output_length = output_length
        self.time_table = self.read_txt()
        self.interval = interval

    def __len__(self):
        return len(self.value_list["AIWSRPF_"])

    def __getitem__(self, idx):
        data_every_moment = {}
        start_time = self.value_list["AIWSRPF_"][idx].split('/')[-1].split('_')[-2]
        time_length = self.input_length+self.output_length
        sequence_date = self.generate_sequence(start_time, time_length, self.interval)
        file_list = {"AIWSRPF_": [], "AIWSRRF_": [], "AIWSRRF-ISO_": [], "AIWSRTF_": [], "AIWSRTF-ISO_": [],
                          "AIWSRUF_": [], "AIWSRUF-ISO_": [], "AIWSRVF_": [], "AIWSRVF-ISO_": []}

        for k in self.value_list.keys():
            file_list[k] = self.generate_file_root(sequence_date, self.value_list[k][idx])

        start = time.time()
        for i in range(1, time_length+1):
            temp = 0
            for j in range(1, self.interval+1):
                temp += np.load((file_list["AIWSRPF_"][i*3-j]))[0]
            data_every_moment[sequence_date[i * 3 - 1]] = [temp]
            for j, k in enumerate(self.value_list.keys()):
                if j % 2 == 0:
                    continue
                temp = np.load(file_list[k][i])
                if temp.shape.__len__() == 3:
                    temp.resize(1, temp.shape[0], temp.shape[1], temp.shape[2])
                temp1 = np.load(file_list[k.split('_')[0]+'-ISO_'][i])
                temp2 = np.concatenate((temp1[:, 5:6], temp1[:, 8:9], temp1[:, 14:15], temp1[:, 22:23]), axis=1)
                data_every_moment[sequence_date[i * 3 - 1]].append(np.concatenate((temp, temp2), axis=1)[0])
        self.transforms(*data_every_moment.values())
        # print(time.time() - start)

        return tuple(torch.cat(temp[1:], dim=0) for temp in list(data_every_moment.values())[0:self.input_length]),\
               [self.classify3h(temp[0][0]) for temp in list(data_every_moment.values())[self.input_length:]]

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
        self.value_list = {"AIWSRPF_":[], "AIWSRRF_":[], "AIWSRRF-ISO_":[], "AIWSRTF_":[], "AIWSRTF-ISO_":[],
                          "AIWSRUF_":[], "AIWSRUF-ISO_":[], "AIWSRVF_":[], "AIWSRVF-ISO_":[]}
        for data in data_list:
            for k in self.value_list.keys():
                if k in data:
                    self.value_list[k].append(data)

        for k in self.value_list.keys():
            self.value_list[k].sort()

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
        RandomCrop(320),
        Normalize(),
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
    with open('LAPS/effective_val.txt') as f:
        for i in f:
            label_files.append(i.split('\n')[0])

    dataset = WData(label_files, transform_train)
    dataloader = data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=8)

    start = time.time()
    statics = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, -1: 0}
    for i, (input, input_next) in enumerate(dataloader):
        input = [temp.cuda() for temp in input]
        input_next = [temp.cuda() for temp in input_next]
        # temp = torch.stack(input_next, dim=0)
        # for j in statics.keys():
        #     statics[j] = (temp == j).sum()
        print(i)
    print(time.time()-start)
    # 555 num_workers=8 without cuda with pin_memory=False
    # 591 num_workers=8 without cuda with pin_memory=True
    # 703 num_workers=16 without cuda with pin_memory=False
    # 801 num_workers=16 without cuda with pin_memory=True

    # for k in statics.keys():
    #     print(statics[k])

