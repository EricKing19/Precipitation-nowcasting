# from netCDF4 import Dataset
import numpy as np
# import h5py
import os
# import pathlib
#
# root = "/home/zhangxinbang/datasets/20170609/test_dir/"
# for _, _, label_files in os.walk(root):
#     continue
#
# root_exist = "/home/jinqizhao2/20170609/Sequence_only_rain_1e-3/"
# for _, _, label_files_exist in os.walk(root_exist):
#     continue
#
# for i in label_files:
#     path = pathlib.Path("/home/jinqizhao2/20170609/Sequence_only_rain_1e-3/"+i.split('.')[0]+'.hdf5')
#     if not path.exists():
#         dataset = Dataset(root+i)
#         precipitation = dataset['Total_precipitation_surface'][0].data.astype(np.float)
#         tem_data = dataset['v-component_of_wind_height_above_ground'][0].data.astype(np.float32)
#         u_data = dataset['v-component_of_wind_height_above_ground'][0].data.astype(np.float32)
#         v_data = dataset['v-component_of_wind_height_above_ground'][0].data.astype(np.float32)
#         humidity_data = dataset['Relative_humidity_height_above_ground'][0].data.astype(np.float32)
#
#         f = h5py.File("/home/jinqizhao2/20170609/Sequence_only_rain_1e-3/"+i.split('.')[0]+'.hdf5', "w")
#         f["precipitation"] = precipitation
#         f["tem"] = tem_data
#         f["u"] = u_data
#         f["v"] = v_data
#         f["humitity"] = humidity_data
#         f.close()
#         print(i)
#


def timer(start, len):
    time_table = [start]
    for i in range(len):
        if str(int(start)+100)[-4:] != '2400':
            time_table.append(str(int(start)+100))
            start = str(int(start) + 100)
        else:
            year = str(start)[0:4]
            month = str(start)[4:6]
            date = str(start)[6:8]
            time = "0000"

            if int(year) % 4 !=0:
                month_table = \
                    {"01":"31", "02":"28", "03":"31", "04":"30", "05":"31", "06":"30", "07":"31", "08":"31", "09":"30", "10":"31", "11":"30", "12":"31"}
            else:
                month_table = \
                    {"01":"31", "02":"29", "03":"31", "04":"30", "05":"31", "06":"30", "07":"31", "08":"31", "09":"30", "10":"31", "11":"30", "12":"31"}

            if month_table[month] != date:
                date = str(int(date) + 1)
                date = date.zfill(2)
            else:
                if month != "12":
                    month = str(int(month) + 1)
                    month = month.zfill(2)
                    date = "01"
                else:
                    year = str(int(year) + 1)
                    month = "01"
                    date = "01"

            start = year + month + date + time
            time_table.append(start)

    with open('time_table.txt','w') as f:
        for i in time_table:
            f.write(i+'\n')
    f.close()

    return time_table


def effective_sample(label_root="/home/zhangxinbang/datasets/weather_dataset_npy/", start="201907010000", len=37*24, effective_len=24):
    '''Added by Tsingzao'''
    # if os.path.exists('./dataset/temp.npy'):
    #     effective_samples_root = np.load('./dataset/temp.npy')
    #     return effective_samples_root.tolist()
    label_files = []
    for root, _, files in os.walk(label_root):
        for file in files:
            if 'CCN' in file:
                label_files.append(os.path.join(root, file))
    label_files.sort()
    # for label_file in label_files:
    # print(label_files)
    # print(label_files.__len__())

    time_table = timer(start, len)
    # print(time_table.__len__())

    effective_samples = []

    for i, t in enumerate(time_table):
        if i > len-effective_len:
            break
        counter = [0 for i in range(effective_len)]
        for j in label_files:
            for l in range(effective_len):
                if time_table[i+l] in j:
                    counter[l] += 1
        if sum(counter) == effective_len * 11:
            effective_samples.append(t)
            print(i)
    # print(effective_samples.__len__())

    effective_samples_root = []

    for s in label_files:
        if s.split('/')[-1].split('_')[-2] in effective_samples:# and 'CCN' in s:
            effective_samples_root.append(s)
            # print(s)
    # print(effective_samples_root.__len__())
    '''Added by Tsingzao'''
    # np.save('./dataset/temp.npy', effective_samples_root)
    return effective_samples_root


if __name__ == '__main__':
    label_root = "/home/zhangxinbang/datasets/weather_dataset_npy_save/2020"
    label_files = effective_sample(label_root, "202002010000", 180*24)

    with open('effective_label.txt', 'w') as f:
        for i in label_files:
            f.write(i+'\n')
    f.close()
