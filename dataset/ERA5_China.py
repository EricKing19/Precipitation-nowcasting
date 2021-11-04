from torch.utils import data
import xarray as xr
import numpy as np
import time


def mean_std(ds, var, levels):
    data = []
    data.append(ds[var].sel(level=levels))

    data = xr.concat(data, 'level').transpose('time', 'level', 'latitude', 'longitude')
    # mean = data.mean(('time', 'latitude', 'longitude')).compute()
    std = data.std('time').mean(('latitude', 'longitude')).compute()

    # mean.name = 'mean'
    std.name = 'std'

    # mean.to_netcdf('mean_{}.nc'.format(var))
    std.to_netcdf('std_{}.nc'.format(var))


class DataGenerator(data.Dataset):
    def __init__(self, ds, var_dict, lead_time, batch_size=32, shuffle=True, load=True, mean=None, std=None, in_len=1, out_len=1):
        """
        Data generator for WeatherBench data.
        Template from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        Args:
            ds: Dataset containing all variables
            var_dict: Dictionary of the form {'var': level}. Use None for level if data is of single level
            lead_time: Lead time in hours
            batch_size: Batch size
            shuffle: bool. If True, data is shuffled.
            load: bool. If True, datadet is loaded into RAM.
            mean: If None, compute mean from data.
            std: If None, compute standard deviation from data.
        """

        self.ds = ds
        self.var_dict = var_dict
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.lead_time = lead_time
        self.in_len = in_len
        self.out_len = out_len

        data = []
        generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
        for var, levels in var_dict.items():
            try:
                data.append(ds[var][var].sel(level=levels))
            except ValueError:
                data.append(ds[var][var].expand_dims({'level': generic_level}, 1))

        self.data = xr.concat(data, 'level').transpose('time', 'level', 'latitude', 'longitude')
        self.mean = xr.open_mfdataset('/home/jinqizhao2/PycharmProjects/Sequence_Precipitation/dataset/mean.nc')['mean'] if mean is None else mean
        self.std = xr.open_mfdataset('/home/jinqizhao2/PycharmProjects/Sequence_Precipitation/dataset/std.nc')['std'] if std is None else std
        # Normalize
        self.data = (self.data - self.mean) / self.std
        self.n_samples = self.data.isel(time=slice(0, -lead_time)).shape[0]
        self.init_time = self.data.isel(time=slice(None, -lead_time)).time
        self.valid_time = self.data.isel(time=slice(lead_time, None)).time

        self.precipitation = xr.open_mfdataset('/home/zhangxinbang/datasets/new_data_2015-2020/rain*', combine='by_coords')
        self.precipitation = self.precipitation.transpose('time', 'level', 'latitude', 'longitude')

        # For some weird reason calling .load() earlier messes up the mean and std computations
        if load:
            print('Loading data into RAM')
            self.data.load()
            self.precipitation.load()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.n_samples

    def __getitem__(self, i):
        'Generate one batch of data'
        if i == 0:
            self.on_epoch_end()
            print('Shuffle Finished')
        # idxs = self.idxs[i * self.batch_size:(i + 1) * self.batch_size]
        idxs = self.idxs[i]
        X, y = [], []
        for i in range(self.in_len):
            X.append(self.data.isel(time=idxs+i).values)
        for j in range(self.out_len):
            y.append(self.precipitation['tp'].isel(time=idxs+self.in_len+j).values)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.idxs = np.arange(self.n_samples)
        if self.shuffle == True:
            np.random.shuffle(self.idxs)


if __name__ == '__main__':
    r_data = xr.open_mfdataset('/home/zhangxinbang/datasets/new_data_2015-2020/h*.nc', combine='by_coords')
    t_data = xr.open_mfdataset('/home/zhangxinbang/datasets/new_data_2015-2020/t*.nc', combine='by_coords')
    u_data = xr.open_mfdataset('/home/zhangxinbang/datasets/new_data_2015-2020/u*.nc', combine='by_coords')
    v_data = xr.open_mfdataset('/home/zhangxinbang/datasets/new_data_2015-2020/v*.nc', combine='by_coords')
    data = {'r': r_data, 't': t_data, 'u': u_data, 'v': v_data}
    var_dict = {'r': [500, 850, 1000], 't': [500, 850, 1000], 'u': [500, 850, 1000], 'v': [500, 850, 1000]}
    # for k in data.keys():
    #     mean_std(data[k], k, var_dict[k])
    lead_time = 5*24

    start = time.time()
    dataset = DataGenerator(data, var_dict, lead_time)
    print(time.time()-start)
    start = time.time()
    for i in range(len(dataset)):
        input, output = dataset[i]
        if i > 10:
            print(time.time() - start)
            break
