from utils.scip_models import maxcut_mccormic_model, CSBaselineSepa, set_aggresive_separation
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from utils.functions import get_normalized_areas
from tqdm import tqdm
import pickle
import pandas as pd
from experiments.cut_selection_dqn.default_parser import parser, get_hparams


# randomize graphs
graph_sizes = [70, 100]
if True:
    graphs = {size: [nx.barabasi_albert_graph(n=size, m=10, seed=223) for _ in range(10)] for size in graph_sizes}
    for glist in graphs.values():
        for g in glist:
            nx.set_edge_attributes(g, {e: np.random.random() for e in g.edges}, 'weight')
        for g1, g2 in combinations(glist, 2):
            assert not nx.is_isomorphic(g1, g2, edge_match=lambda e1, e2: e1['weight'] == e2['weight'])

    with open('maxcut-variability-graphs.pkl', 'wb') as f:
        pickle.dump(graphs, f)

with open('maxcut-variability-graphs.pkl', 'rb') as f:
    graphs = pickle.load(f)

if True:
    results = {k: {size: [] for size in graph_sizes} for k in ['default', 'aggressive']}
    seeds = [46, 72, 101]
    lp_iterations_limit = 2000
    for policy in ['default', 'aggressive']:
        for size, glist in graphs.items():
            for G in tqdm(glist, desc=f'solving size={size} policy={policy}'):
                model, x, cycle_sepa = maxcut_mccormic_model(G, use_random_branching=False)
                model.hideOutput(True)
                model.optimize()
                optval = model.getObjVal()
                assert model.getGap() == 0

                for seed in seeds:
                    model, x, cycle_sepa = maxcut_mccormic_model(G)
                    sepa = CSBaselineSepa(hparams={'lp_iterations_limit': lp_iterations_limit})
                    model.includeSepa(sepa, '#CS_baseline', 'do-nothing', priority=-100000000, freq=1)
                    model.setBoolParam("misc/allowdualreds", 0)

                    if policy == 'aggressive':
                        set_aggresive_separation(model)  # todo debug
                    model.setLongintParam('limits/nodes', 1)  # solve only at the root node
                    model.setIntParam('separating/maxstallroundsroot', -1)  # add cuts forever
                    model.setIntParam('branching/random/priority', 10000000)
                    model.setBoolParam('randomization/permutevars', True)
                    model.setIntParam('randomization/permutationseed', seed)
                    model.setIntParam('randomization/randomseedshift', seed)
                    model.hideOutput(True)
                    model.optimize()
                    sepa.update_stats()
                    stats = sepa.stats
                    db, gap, lpiter = stats['dualbound'], stats['gap'], stats['lp_iterations']
                    ncuts, napplied_cumsum = np.array(stats['ncuts'][1:]), np.array(stats['ncuts_applied'])
                    db_auc = get_normalized_areas(t=lpiter, ft=db, t_support=lp_iterations_limit, reference=optval)
                    gap_auc = get_normalized_areas(t=lpiter, ft=gap, t_support=lp_iterations_limit, reference=0)
                    napplied_round = napplied_cumsum[1:] - napplied_cumsum[:-1]
                    applied_avail = napplied_round / ncuts
                    stats['db_auc'] = db_auc
                    stats['gap_auc'] = gap_auc
                    stats['optval'] = optval
                    stats['napplied/round'] = np.mean(napplied_round)
                    stats['applied/avail'] = np.mean(applied_avail)
                    stats['napplied_std'] = np.std(napplied_round)
                    stats['applied/avail_std'] = np.std(applied_avail)


                    results[policy][size].append(stats)

    with open('maxcut-variability-results.pkl', 'wb') as f:
        pickle.dump(results, f)

with open('maxcut-variability-results.pkl', 'rb') as f:
    results = pickle.load(f)

dfs = {}
for size in graph_sizes:
    summary = {}
    columns = ['db_auc', 'gap_auc', 'gap', 'time', 'rounds', 'total ncuts', 'napplied/round', 'applied/avail']
    for policy in ['default', 'aggressive']:
        db_auc_list = np.array([s['db_auc'] for s in results[policy][size]])
        gap_auc_list = np.array([s['gap_auc'] for s in results[policy][size]])
        gap = np.array([s['gap'][-1] for s in results[policy][size]])
        time = np.array([s['solving_time'][-1] for s in results[policy][size]])
        ncuts = np.array([sum(s['ncuts']) for s in results[policy][size]])
        napplied_round = np.array([s['napplied/round'] for s in results[policy][size]])
        applied_avail = np.array([s['applied/avail'] for s in results[policy][size]])
        napplied_std = np.array([s['napplied_std'] for s in results[policy][size]])
        applied_avail_std = np.array([s['applied/avail_std'] for s in results[policy][size]])
        nrounds = np.array([s['lp_rounds'][-1] for s in results[policy][size]])
        summary[policy] = ['{:.4f} ({:.4f})'.format(arr.mean(), arr.std()) for arr in [db_auc_list, gap_auc_list, gap, time, nrounds, ncuts]]
        summary[policy] += ['{:.4f} ({:.4f})'.format(napplied_round.mean(), napplied_std.mean())]
        summary[policy] += ['{:.4f} ({:.4f})'.format(applied_avail.mean(), applied_avail_std.mean())]
    df = pd.DataFrame.from_dict(summary, orient='index', columns=columns)
    print('#'*30 + f' SIZE {size} ' + '#'*30)
    print(df)
    dfs[size] = df

with open('maxcut-variability-dfs.pkl', 'wb') as f:
    pickle.dump(dfs, f)




if False:
    G = nx.barabasi_albert_graph(n=150, m=10, seed=223)
    nx.set_node_attributes(G, {i: np.random.random() for i in G.nodes}, 'c')
    fig, axes = plt.subplots(2, 2)

    for aggresive in [True, False]:
        model, x = mvc_model(G, use_general_cuts=True)
        sepa = BaselineSepa(hparams={'lp_iterations_limit': 1500})
        model.includeSepa(sepa, '#CS_baseline', 'do-nothing', priority=-100000000, freq=1)
        model.setBoolParam("misc/allowdualreds", 0)

        if aggresive:
            set_aggresive_separation(model)  # todo debug
        model.setLongintParam('limits/nodes', 1)  # solve only at the root node
        model.setIntParam('separating/maxstallroundsroot', -1)  # add cuts forever
        model.setIntParam('branching/random/priority', 10000000)  # add cuts forever

        model.setBoolParam('randomization/permutevars', True)
        model.setIntParam('randomization/permutationseed', 223)
        model.setIntParam('randomization/randomseedshift', 223)

        model.optimize()

        sepa.update_stats()
        # plt.figure()
        # nx.spring_layout(G)
        # nx.draw(G, labels={i: model.getVal(x[i]) for i in G.nodes}, with_labels=True, node_color=['gray' if model.getVal(x[i]) == 0 else 'blue' for i in G.nodes])
        # plt.savefig('test-sol.png')
        axes[0,0].plot(sepa.stats['lp_iterations'], sepa.stats['dualbound'], label='aggressive' if aggresive else 'default')
        axes[0,1].plot(sepa.stats['lp_iterations'], sepa.stats['gap'], label='aggressive' if aggresive else 'default')
        axes[1,0].plot(sepa.stats['lp_iterations'], sepa.stats['solving_time'], label='aggressive' if aggresive else 'default')
        axes[1,1].plot(sepa.stats['solving_time'], sepa.stats['gap'], label='aggressive' if aggresive else 'default')

    # axes[0, 0].set_title('default')
    # axes[2, 0].set_xlabel('lp iter')
    # axes[2, 1].set_xlabel('lp iter')
    # axes[0, 1].set_title('aggressive')
    axes[0,0].set_title('db')
    axes[0,1].set_title('gap')
    axes[1,0].set_title('time')
    axes[1,1].set_title('gap')
    axes[0,0].set_xlabel('lp iterations')
    axes[0,1].set_xlabel('lp iterations')
    axes[1,0].set_xlabel('lp iterations')
    axes[1,1].set_xlabel('time')
    axes[0,0].legend()
    axes[1,0].legend()
    axes[1,0].legend()
    axes[1,1].legend()

    plt.tight_layout()
    fig.savefig('test-aggressive-db-gap-time.png')

print('finish')