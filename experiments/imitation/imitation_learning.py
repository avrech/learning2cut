# copied from ogb repository

from sklearn.metrics import roc_auc_score
import os
from experiments.imitation.cutting_planes_dataset import CuttingPlanesDataset

import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import argparse
import numpy as np
# from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

from experiments.imitation.evaluator import Evaluator
from gnn.models import CutSelectionModel

criterion = torch.nn.BCELoss()


def train(model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()

            loss = criterion(pred.to(torch.float32), batch.y.view(-1, ))

            loss.backward()
            optimizer.step()


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(-1, 1).detach().cpu())
            y_pred.append(torch.argmax(pred.detach(), dim=1).view(-1, 1).cpu())

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
    parser.add_argument('--emb_dim', type=int, default=300,
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
    parser.add_argument('--datadir', type=str, default='data/barabasi-albert-n50-m10-weights-normal-seed36/examples',
                        help='path to load data from')
    parser.add_argument('--lr', type=int, default=1e-3,
                        help='SGD learning rate')

    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting

    # load dataset
    dataset = CuttingPlanesDataset(args.datadir, savefile=False)

    # loader = DataLoader(dataset, batchsize=args.batch_size, follow_batch=['x_s', 'x_t'])
    #
    # dataset = dataset.shuffle()
    # one_tenth_length = int(len(dataset) * 0.1)
    # train_dataset = dataset[:one_tenth_length * 8]
    # val_dataset = dataset[one_tenth_length * 8:one_tenth_length * 9]
    # test_dataset = dataset[one_tenth_length * 9:]
    # len(train_dataset), len(val_dataset), len(test_dataset)

    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    # if args.gnn == 'gin':
    #     model = GNN(gnn_type='gin', num_class=dataset.num_classes, emb_dim=args.emb_dim, drop_ratio=args.drop_ratio,
    #                 virtual_node=False).to(device)
    # elif args.gnn == 'gin-virtual':
    #     model = GNN(gnn_type='gin', num_class=dataset.num_classes, emb_dim=args.emb_dim, drop_ratio=args.drop_ratio,
    #                 virtual_node=True).to(device)
    # elif args.gnn == 'gcn':
    #     model = GNN(gnn_type='gcn', num_class=dataset.num_classes, emb_dim=args.emb_dim, drop_ratio=args.drop_ratio,
    #                 virtual_node=False).to(device)
    # elif args.gnn == 'gcn-virtual':
    #     model = GNN(gnn_type='gcn', num_class=dataset.num_classes, emb_dim=args.emb_dim, drop_ratio=args.drop_ratio,
    #                 virtual_node=True).to(device)
    # else:
    #     raise ValueError('Invalid GNN type')

    # model
    hparams = vars(args)
    model = CutSelectionModel(hparams=hparams)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    valid_curve = []
    test_curve = []
    train_curve = []

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train(model, device, train_loader, optimizer)

        print('Evaluating...')
        train_perf = eval(model, device, train_loader, evaluator)
        valid_perf = eval(model, device, valid_loader, evaluator)
        test_perf = eval(model, device, test_loader, evaluator)

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        train_curve.append(train_perf['acc'])
        valid_curve.append(valid_perf['acc'])
        test_curve.append(test_perf['acc'])

    best_val_epoch = np.argmax(np.array(valid_curve))
    best_train = max(train_curve)

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

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