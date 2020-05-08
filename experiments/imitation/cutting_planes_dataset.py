import torch
from torch_geometric.data import InMemoryDataset
import os
from tqdm import tqdm
import pickle
from utils.data import get_pair_data_state, get_pair_data_memory

class CuttingPlanesDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, hparams={}, savefile=False):
        self.hparams = hparams
        self.savefile = savefile
        self.filename = hparams.get('filename', 'cutting_planes_dataset.pt')
        self.num_instances = 0
        self.num_examples = 0
        super(CuttingPlanesDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        with open(self.processed_paths[0][:-3] + '_stats.pkl', 'rb') as f:
            stats = pickle.load(f)
        self.num_instances = stats['num_instances']
        self.num_examples = stats['num_examples']

        if not self.savefile:
            # delete the generated dataset file.
            os.remove(self.processed_paths[0])
            os.remove(self.processed_paths[0][:-3] + '_stats.pkl')
            print('removed auto-generated dataset files from file system ')

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
            if rawfile[-14:] == 'scip_state.pkl':
                with open(os.path.join(self.root, rawfile), 'rb') as f:
                    state_action_list = pickle.load(f)
                for s, a in state_action_list:
                    pairdata = get_pair_data_state(s, scip_action=a)
                    data_list.append(pairdata)
                    self.num_examples += 1
                self.num_instances += 1

        # TODO: filter redundant features, preprocess, normalize etc.
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        with open(self.processed_paths[0][:-3] + '_stats.pkl', 'wb') as f:
            pickle.dump({'num_instances': self.num_instances,
                         'num_examples': self.num_examples}, f)

    def size(self, units='G'):
        return get_pair_data_memory(self.data, units=units)

    @property
    def num_classes(self):
        r"""The number of classes in the dataset."""
        data = self.data
        return data.y_t.max().item() + 1 if data.y_t.dim() == 1 else data.y_t.size(1)

if __name__ == '__main__':
    rootdir = 'data/barabasi-albert-n50-m10-weights-normal-seed36/examples'
    dataset = CuttingPlanesDataset(rootdir)
    from utils.functions import get_data_memory

    print('Such dataset of 1000 instances will:')
    print('provide ', int(dataset.num_examples*1000/dataset.num_instances), ' examples')
    print('consume ', '{:.2f}'.format(get_pair_data_memory(dataset.data, 'G')*1000/dataset.num_instances), 'Gbytes')
    print('finished')