import torch
from torch_geometric.data import InMemoryDataset
import os
from tqdm import tqdm
import pickle


class CuttingPlanesDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, hparams={}, savefile=True):
        self.hparams = hparams
        self.savefile = savefile

        self.filename = hparams.get('filename', 'cutting_planes_dataset.pt')
        self.num_instances = 0
        self.num_examples = 0
        super(CuttingPlanesDataset, self).__init__(root, transform, pre_transform)
        if self.savefile:
            self.data, self.slices = torch.load(self.processed_paths[0])
            with open(self.processed_paths[0][:-3] + '_stats.pkl', 'rb') as f:
                stats = pickle.load(f)
            self.num_instances = stats['num_instances']
            self.num_examples = stats['num_examples']

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [self.filename]

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        rawfiles = os.listdir(self.root)
        # assuming each file contains a list of Data objects
        for rawfile in tqdm(rawfiles, desc='Loading raw files'):
            if rawfile[-3:] == '.pt':
                data = torch.load(os.path.join(self.root, rawfile))
                self.num_examples += len(data)
                self.num_instances += 1
                data_list += data

        # TODO: filter redundant features, preprocess, normalize etc.
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        if self.savefile:
            torch.save((data, slices), self.processed_paths[0])
            with open(self.processed_paths[0][:-3] + '_stats.pkl', 'wb') as f:
                pickle.dump({'num_instances': self.num_instances,
                             'num_examples': self.num_examples}, f)
        else:
            self.data = data
            self.slices = slices

if __name__ == '__main__':
    rootdir = 'data/barabasi-albert-n50-m10-weights-normal-seed36/examples'
    dataset = CuttingPlanesDataset(rootdir)
    from utils.functions import get_data_memory

    print('Such dataset of 1000 instances will:')
    print('provide ', int(dataset.num_examples*1000/dataset.num_instances), ' examples')
    print('consume ', '{:.2f}'.format(get_data_memory(dataset.data, 'G')*1000/dataset.num_instances), 'Gbytes')
    print('finished')