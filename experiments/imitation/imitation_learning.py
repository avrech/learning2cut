# copied from ogb repository

from experiments.imitation.cutting_planes_dataset import CuttingPlanesDataset

import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
import argparse
import numpy as np

from experiments.imitation.evaluator import Evaluator
from gnn.models import CutSelectionModel
import torch.utils.tensorboard as tb
criterion = torch.nn.BCELoss()


def train(model, device, loader, optimizer):
    model.train()

    for batch in loader:
        batch = batch.to(device)

        if batch.x_c.shape[0] == 1 or batch.x_c_batch[-1] == 0: #  TODO what is this for?
            pass
        else:
            pred, probs = model(batch)
            optimizer.zero_grad()

            loss = criterion(probs.to(torch.float32), batch.y.view(-1, 1))

            loss.backward()
            optimizer.step()


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for batch in loader:
        batch = batch.to(device)

        if batch.x_c.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred, probs = model(batch)

            y_true.append(batch.y.view(-1, 1).detach().cpu())
            y_pred.append(pred.detach().view(-1, 1).cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


# def add_zeros(data):
#     data.x = torch.zeros(data.num_nodes, dtype=torch.long)
#     return data


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbg-ppa data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=32,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="cutting-planes-dataset",
                        help='dataset name (default: cutting-planes-dataset)')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')


    # cutting-planes stuff
    parser.add_argument('--logdir', type=str, default='results/test',
                        help='path to results root')
    parser.add_argument('--datadir', type=str, default='data/barabasi-albert-n15-m7-weights-normal-seed36/examples',
                        help='path to load data from')
    parser.add_argument('--lr', type=int, default=1e-3,
                        help='SGD learning rate')
    parser.add_argument('--cuts_embedding_layers', type=int, default=1,
                        help='number of CutsEmbedding modules sequentialized')
    parser.add_argument('--cuts_embedding_aggr', type=str, default='mean',
                        help='aggregation function for message passing')
    parser.add_argument('--factorization_arch', type=str, default='FGConv',
                        help='FGConv or GraphUnet only')
    parser.add_argument('--factorization_aggr', type=str, default='mean',
                        help='aggregation function for message passing')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting

    # load dataset
    dataset = CuttingPlanesDataset(args.datadir, savefile=False)

    # loader = DataLoader(dataset, batchsize=args.batch_size, follow_batch=['x_s', 'x_t'])

    dataset = dataset.shuffle()
    one_tenth_length = int(len(dataset) * 0.1)
    train_dataset = dataset[:one_tenth_length * 8]
    val_dataset = dataset[one_tenth_length * 8:one_tenth_length * 9]
    test_dataset = dataset[one_tenth_length * 9:]
    print(f'Train-set: {len(train_dataset)}, Validation-set: {len(val_dataset)}, Test-set: {len(test_dataset)}')

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, follow_batch=['x_c', 'x_v', 'x_a'])
    valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, follow_batch=['x_c', 'x_v', 'x_a'])
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, follow_batch=['x_c', 'x_v', 'x_a'])

    # model
    hparams = vars(args)
    hparams['state_x_c_channels'] = train_dataset.data.x_c.shape[-1]
    hparams['state_x_v_channels'] = train_dataset.data.x_v.shape[-1]
    hparams['state_x_a_channels'] = train_dataset.data.x_a.shape[-1]
    hparams['state_edge_attr_dim'] = 1  # TODO: write in a more general way

    model = CutSelectionModel(hparams=hparams)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    valid_curve = []
    test_curve = []
    train_curve = []

    writer = tb.SummaryWriter(log_dir=args.logdir)

    for epoch in range(1, args.epochs + 1):
        # print("=====Epoch {}".format(epoch))
        # print('Training...')
        train(model, device, train_loader, optimizer)

        # print('Evaluating...')
        train_perf = eval(model, device, train_loader, evaluator)
        valid_perf = eval(model, device, valid_loader, evaluator)
        test_perf = eval(model, device, test_loader, evaluator)

        print("Epoch {}".format(epoch),
              '\t| Train: ', ['{}: {:.3f}'.format(k, v) for k, v in train_perf.items()],
              '\t| Validation: ', ['{}: {:.3f}'.format(k, v) for k, v in valid_perf.items()],
              '\t| Test: ', ['{}: {:.3f}'.format(k, v) for k, v in test_perf.items()])

        tb_scalars = {f'{k}/{s}': v
                      for s, perf in zip(['Train', 'Validation', 'Test'], [train_perf, valid_perf, test_perf])
                      for k, v in perf.items()}
        for k, v in tb_scalars.items():
            writer.add_scalar(k, v, epoch)

        train_curve.append(train_perf['acc'])
        valid_curve.append(valid_perf['acc'])
        test_curve.append(test_perf['acc'])

    best_val_epoch = np.argmax(np.array(valid_curve))
    best_train = max(train_curve)

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

    writer.add_hparams(hparam_dict=hparams,
                       metric_dict={'best_valid': valid_curve[best_val_epoch],
                                    'test_score': test_curve[best_val_epoch],
                                    'best_train': best_train})
    if not args.filename == '':
        torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch],
                    'Train': train_curve[best_val_epoch], 'BestTrain': best_train}, args.filename)


if __name__ == "__main__":
    main()







# from torch_geometric.data import DataLoader
# batch_size= 512
# train_loader = DataLoader(train_dataset, batch_size=batch_size)
# val_loader = DataLoader(val_dataset, batch_size=batch_size)
# test_loader = DataLoader(test_dataset, batch_size=batch_size)
#
#
# device = torch.device('cuda')
# model = Net().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# crit = torch.nn.BCELoss()
#
#
# def train():
#     model.train()
#
#     loss_all = 0
#     for data in train_loader:
#         data = data.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#
#         label = data.y.to(device)
#         loss = crit(output, label)
#         loss.backward()
#         loss_all += data.num_graphs * loss.item()
#         optimizer.step()
#     return loss_all / len(train_dataset)
#
#
#
#
# def evaluate(loader):
#     model.eval()
#
#     predictions = []
#     labels = []
#
#     with torch.no_grad():
#         for data in loader:
#             data = data.to(device)
#             pred = model(data).detach().cpu().numpy()
#
#             label = data.y.detach().cpu().numpy()
#             predictions.append(pred)
#             labels.append(label)
#
#     predictions = np.hstack(predictions)
#     labels = np.hstack(labels)
#
#     return roc_auc_score(labels, predictions)
#
#
# for epoch in range(1, 200):
#     loss = train()
#     train_acc = evaluate(train_loader)
#     val_acc = evaluate(val_loader)
#     test_acc = evaluate(test_loader)
#     print('Epoch: {:03d}, Loss: {:.5f}, Train Auc: {:.5f}, Val Auc: {:.5f}, Test Auc: {:.5f}'.
#           format(epoch, loss, train_acc, val_acc, test_acc))