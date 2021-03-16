# distributedRL base class for learner - implements the distributed part
import time
from collections import deque
import pyarrow as pa
import zmq
import threading
from time import time
import numpy as np
from utils.data import Transition
from utils.misc import get_img_from_fig
import os
from gnn.models import Qnet, TQnet, TransformerDecoderContext
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max, scatter_add
from collections import namedtuple
import matplotlib as mpl
import pickle
mpl.rc('figure', max_open_warning=0)
import wandb


StateActionContext = namedtuple('StateActionQValuesContext', ('scip_state', 'action', 'q_values', 'transformer_context'))
DemonstrationBatch = namedtuple('DemonstrationBatch', (
    'context_edge_index',
    'context_edge_attr',
    'action',
    'idx',
    'conv_aggr_out_idx',
    'encoding_broadcast',
    'action_batch',
))


class DQNLearner:
    """
    This actor executes in parallel two tasks:
        a. receiving batches, sending priorities and publishing params.
        b. optimizing the model.
    According to
    https://stackoverflow.com/questions/54937456/how-to-make-an-actor-do-two-things-simultaneously
    Ray's Actor.remote() cannot execute to tasks in parallel. We adopted the suggested solution in the
    link above, and run the IO in a background process on Actor instantiation.
    The main Actor's thread will optimize the model using the GPU.
    """
    def __init__(self, hparams, use_gpu=True, gpu_id=None, run_io=False, run_setup=False, **kwargs):
        self.hparams = hparams

        cuda_id = 'cuda' if gpu_id is None else f'cuda:{gpu_id}'
        self.device = torch.device(cuda_id if use_gpu and torch.cuda.is_available() else "cpu")
        self.batch_size = hparams.get('batch_size', 64)
        self.gamma = hparams.get('gamma', 0.999)
        if hparams.get('dqn_arch', 'TQNet'):
            # todo - consider support also mean value aggregation.
            assert hparams.get('value_aggr') == 'max', "TQNet v3 supports only value_aggr == max"
            assert hparams.get('tqnet_version',
                               'v3') == 'v3', 'v1 and v2 are no longer supported. need to adapt to new decoder context'
        self.policy_net = TQnet(hparams=hparams, use_gpu=use_gpu, gpu_id=gpu_id).to(self.device) if hparams.get(
            'dqn_arch', 'TQNet') == 'TQNet' else Qnet(hparams=hparams).to(self.device)
        self.target_net = TQnet(hparams=hparams, use_gpu=use_gpu, gpu_id=gpu_id).to(self.device) if hparams.get(
            'dqn_arch', 'TQNet') == 'TQNet' else Qnet(hparams=hparams).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.tqnet_version = hparams.get('tqnet_version', 'v3')
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=hparams.get('lr', 0.001),
                                    weight_decay=hparams.get('weight_decay', 0.0001))
        # value aggregation method for the target Q values
        if hparams.get('value_aggr', 'mean') == 'max':
            self.value_aggr = scatter_max
        elif hparams.get('value_aggr', 'mean') == 'mean':
            self.value_aggr = scatter_mean
        self.nstep_learning = hparams.get('nstep_learning', 1)
        self.dqn_objective = hparams.get('dqn_objective', 'db_auc')
        self.use_transformer = hparams.get('dqn_arch', 'TQNet') == 'TQNet'

        # training stuff
        self.num_env_steps_done = 0
        self.num_sgd_steps_done = 0
        self.num_param_updates = 0
        self.training = True
        self.walltime_offset = 0
        self.start_time = time()
        self.last_time_sec = self.walltime_offset

        # logging
        self.is_learner = True
        # file system paths
        # todo - set worker-specific logdir for distributed DQN
        self.run_dir = hparams['run_dir']
        self.checkpoint_filepath = os.path.join(self.run_dir, 'checkpoint.pt')
        # self.writer = SummaryWriter(log_dir=os.path.join(self.run_dir, 'tensorboard'))  # todo remove after wandb works
        self.print_prefix = ''

        self.n_step_loss_moving_avg = 0
        self.demonstration_loss_moving_avg = 0

        # initialize (set seed and load checkpoint)
        self.initialize_training()

        # idle time monitor
        self.idle_time_sec = 0

        # # set learner specific logdir
        # learner_logdir = os.path.join(self.run_dir, 'tensorboard', 'learner')  # todo remove after wandb works
        # self.writer = SummaryWriter(log_dir=learner_logdir)

        # set checkpoint file path for learner and workers
        self.checkpoint_filepath = os.path.join(self.run_dir, 'learner_checkpoint.pt')

        self.replay_data_queue = deque(maxlen=hparams.get('max_pending_requests', 10)+1)
        self.new_priorities_queue = deque(maxlen=hparams.get('max_pending_requests', 10)+1)
        self.new_params_queue = deque(maxlen=hparams.get('max_pending_requests', 10)+1)
        # todo - inherited and recovered in CutDQNAgent
        #  self.num_param_updates = 0

        # number of SGD steps between each workers update
        self.param_sync_interval = hparams.get("param_sync_interval", 50)
        self.print_prefix = '[Learner] '


        # initialize zmq sockets reusing last run communication config
        print(self.print_prefix, "initializing sockets..")

        context = zmq.Context()
        self.learner_2_apex_socket = context.socket(zmq.PUSH)  # for sending logs
        self.replay_server_2_learner_socket = context.socket(zmq.PULL)  # for receiving batch from replay server
        self.learner_2_replay_server_socket = context.socket(zmq.PUSH)  # for sending back new priorities to replay server
        self.params_pub_socket = context.socket(zmq.PUB)  # for publishing new params to workers

        if run_setup:
            # connect to the main apex process
            self.learner_2_apex_socket.connect(f'tcp://127.0.0.1:{hparams["com"]["apex_port"]}')
            self.print(f'connecting to apex_port: {hparams["com"]["apex_port"]}')

            # bind sockets to random free ports
            hparams['com']["replay_server_2_learner_port"] = self.replay_server_2_learner_socket.bind_to_random_port('tcp://127.0.0.1', min_port=10000, max_port=60000)
            hparams['com']["learner_2_workers_pubsub_port"] = self.params_pub_socket.bind_to_random_port('tcp://127.0.0.1', min_port=10000, max_port=60000)
            self.print(f'binding to random ports: replay_server_2_learner_port={hparams["com"]["replay_server_2_learner_port"]}, learner_2_workers_pubsub_port={hparams["com"]["learner_2_workers_pubsub_port"]}')
            # send com config to apex
            message = pa.serialize(('learner_com_cfg', list(hparams['com'].items()))).to_buffer()
            self.learner_2_apex_socket.send(message)
            # wait for replay_server com config
            message = self.replay_server_2_learner_socket.recv()
            topic, body = pa.deserialize(message)
            assert topic == 'replay_server_com_cfg'
            learner_2_replay_server_port = {k: v for k, v in body}['learner_2_replay_server_port']
            hparams['com']['learner_2_replay_server_port'] = learner_2_replay_server_port
            self.print(f'connecting to learner_2_replay_server_port: {hparams["com"]["learner_2_replay_server_port"]}')
            self.learner_2_replay_server_socket.connect(f'tcp://127.0.0.1:{learner_2_replay_server_port}')

        else:
            # reuse com config
            self.print('connecting to ports: ', hparams["com"])
            self.learner_2_apex_socket.connect(f'tcp://127.0.0.1:{hparams["com"]["apex_port"]}')
            self.replay_server_2_learner_socket.bind(f'tcp://127.0.0.1:{hparams["com"]["replay_server_2_learner_port"]}')
            self.learner_2_replay_server_socket.connect(f'tcp://127.0.0.1:{hparams["com"]["learner_2_replay_server_port"]}')
            self.params_pub_socket.bind(f'tcp://127.0.0.1:{hparams["com"]["learner_2_workers_pubsub_port"]}')
            self.print('reusing ports', hparams['com'])

        self.initialize_training()

        # # todo wandb
        # # we must call wandb.init in each process wandb.log is called.
        # # in distributed_unittest we shouldn't do it however.
        # # create a config dict for comparing hparams, grouping and other operations on wandb dashboard
        # wandb_config = hparams.copy()
        # wandb_config.pop('datasets')
        # wandb_config.pop('com')
        # wandb.init(resume='allow',  # hparams['resume'],
        #            id=hparams['run_id'],
        #            project=hparams['project'],
        #            config=wandb_config,
        #            reinit=True  # for distributed_unittest.py
        #            )

        if run_io:
            self.print('running io in background')
            self.background_io = threading.Thread(target=self.run_io, args=())
            self.background_io.start()

        # save pid to run_dir
        pid = os.getpid()
        pid_file = os.path.join(hparams["run_dir"], 'learner_pid.txt')
        self.print(f'saving pid {pid} to {pid_file}')
        with open(pid_file, 'w') as f:
            f.writelines(str(pid) + '\n')

    @staticmethod
    def params_to_numpy(model: torch.nn.Module):
        params = []
        # todo - why deepcopy fails on TypeError: can't pickle torch._C.ScriptFunction objects
        # new_model = deepcopy(model)
        # state_dict = new_model.cpu().state_dict()
        for param in model.state_dict().values():
            params.append(param.cpu().numpy())
        return params

    # todo - this method is unused. remove it
    def get_params_packet(self, packet_id):
        """
        pack the learner params together with unique packet_id,
        which is essentially the self.num_param_updates counter.
        This packet_id will be used to synchronize the test-worker global_step robustly to failures.
        """
        model = self.policy_net
        params = self.params_to_numpy(model)
        params_packet = (params, packet_id)
        params_packet = pa.serialize(params_packet).to_buffer()
        return params_packet

    def publish_params(self):
        if len(self.new_params_queue) > 0:
            params_packet = self.new_params_queue.popleft()  # thread-safe
            # attach a 'topic' to the packet and send
            message = pa.serialize(('new_params', params_packet)).to_buffer()
            self.params_pub_socket.send(message)

    def prepare_new_params_to_workers(self):
        """
        Periodically snapshot the learner policy params,
        and push into new_params_queue with the corresponding num_params_update.
        In addition, checkpoint the model and statistics, to properly recover from failures.
        Both Learner and Worker classes will use this checkpoint for recovering.
        The test-worker instead has its own checkpoint, including the latest log stats and some more stuff.
        The test worker will synchronize to the learner state every params update, so it won't be
        affected from checkpointing separately.
        """
        if self.num_sgd_steps_done > 0 and self.num_sgd_steps_done % self.param_sync_interval == 0:
            self.num_param_updates += 1

            # prepare params_packet
            model = self.policy_net
            params = self.params_to_numpy(model)
            params_packet = (params, int(self.num_param_updates))
            self.new_params_queue.append(params_packet)  # thread-safe

            # log stats here - to be synchronized with the workers and tester logs.
            # todo - if self.num_param_updates > 0 and self.num_param_updates % self.hparams.get('log_interval', 100) == 0:
            cur_time_sec = time.time() - self.start_time + self.walltime_offset
            info = {'Idle time': '{:.2f}%'.format(self.idle_time_sec / (cur_time_sec - self.last_time_sec))}
            global_step, log_dict = self.log_stats(info=info, log_directly=False)
            logs_packet = ('log', 'learner', [('global_step', global_step)] + [(k, v) for k, v in log_dict.items()])
            logs_packet = pa.serialize(logs_packet).to_buffer()
            self.learner_2_apex_socket.send(logs_packet)
            self.save_checkpoint()

    def unpack_batch_packet(self, batch_packet):
        """ Prepares received data for sgd """
        transition_numpy_tuples, weights, idxes, data_ids = pa.deserialize(batch_packet)
        transitions = [Transition.from_numpy_tuple(npt) for npt in transition_numpy_tuples]
        sgd_step_inputs = self.preprocess_batch(transitions, weights, idxes, data_ids)
        return sgd_step_inputs

    def recv_batch(self, blocking=True):
        """
        Receives a batch from replay server.
        Returns True if any batch received, otherwise False
        """
        received = False
        if blocking:
            batch_packet = self.replay_server_2_learner_socket.recv()
            received = True
        else:
            try:
                batch_packet = self.replay_server_2_learner_socket.recv(zmq.DONTWAIT)
                received = True
            except zmq.Again:
                pass
        if received:
            batch = self.unpack_batch_packet(batch_packet)
            self.replay_data_queue.append(batch)
        return received

    def send_new_priorities(self):
        if len(self.new_priorities_queue) > 0:
            new_priorities = self.new_priorities_queue.popleft()
            new_priorities_packet = pa.serialize(new_priorities).to_buffer()
            self.learner_2_replay_server_socket.send(new_priorities_packet)

    # old version
    # def run(self):
    #     self.initialize_training()
    #     time.sleep(3)
    #     while True:
    #         self.recv_batch()  # todo run in background
    #         replay_data = self.replay_data_queue.pop()
    #         new_priorities = self.learning_step(replay_data)
    #         self.new_priorities_queue.append(new_priorities)
    #         self.send_new_priorities()
    #
    #         self.prepare_new_params_to_workers()
    #         self.publish_params()

    def run_io(self):
        """
        asynchronously receive data and return new priorities to replay server,
        and publish new params to workers
        """
        self.print('(background) started io process in background...')
        self.print('(background) sending "restart" message to replay_server...')
        restart_message = pa.serialize("restart").to_buffer()
        self.learner_2_replay_server_socket.send(restart_message)

        time.sleep(2)
        self.print('(background) running io loop')
        while True:
            self.recv_batch(blocking=False)
            self.send_new_priorities()
            self.publish_params()

    def optimize_model(self):
        """
        Overrides GDQN.optimize_model()
        Instead of sampling from the local replay buffer like in the single thread GDQN,
        we pop one batch from replay_data_queue, process and push new priorities to new_priorities_queue.
        Sending those priorities, and updating the workers will be done asynchronously in a separate thread.
        """
        # wait until there is any batch ready for processing, and count the idle time
        idle_time_start = time.time()
        idle_time_end = time.time()
        while not self.replay_data_queue:
            idle_time_end = time.time()
        self.idle_time_sec = idle_time_end - idle_time_start

        # pop one batch and perform one SGD step
        batch, weights, idxes, data_ids, is_demonstration, demonstration_batch = self.replay_data_queue.popleft()  # thread-safe pop

        new_priorities = self.sgd_step(batch=batch, importance_sampling_correction_weights=weights, is_demonstration=is_demonstration, demonstration_batch=demonstration_batch)
        packet = (idxes, new_priorities, data_ids)
        self.new_priorities_queue.append(packet)  # thread safe append
        # todo verify
        if self.num_sgd_steps_done % self.hparams.get('target_update_interval', 1000) == 0:
            self.update_target()

    def run(self):
        """
        asynchronously
        pop batch from replay_data_queue,
        push new priorities to queue
        and periodically push updated params to param_queue
        """
        print(self.print_prefix + 'started main optimization loop')
        time.sleep(1)
        while True:
            self.optimize_model()
            self.prepare_new_params_to_workers()

    def preprocess_batch(self, transitions, weights, idxes, data_ids):
        # sort demonstration transitions first:
        is_demonstration = np.array([t.is_demonstration for t in transitions], dtype=np.bool)
        argsort_demonstrations_first = is_demonstration.argsort()[::-1]
        transitions = [transitions[idx] for idx in argsort_demonstrations_first]
        weights, idxes, data_ids = weights[argsort_demonstrations_first], idxes[argsort_demonstrations_first], data_ids[argsort_demonstrations_first]
        is_demonstration = is_demonstration[argsort_demonstrations_first]

        # create pytorch-geometric batch
        batch = Transition.create_batch(transitions)

        # prepare demonstration data if any
        if is_demonstration.any():
            # filter non demonstration data
            n_demonstrations = sum(is_demonstration)  # number of demonstration transitions in batch
            # remove from edge_index (and corresponding edge_attr) edges which reference cuts of index greater than
            max_demonstration_edge_index = sum(batch.x_a_batch < n_demonstrations) - 1
            demonstration_edges = batch.demonstration_context_edge_index[0, :] < max_demonstration_edge_index

            # demonstration_context_edge_index = batch.demonstration_context_edge_index[:, demonstration_edges]
            # demonstration_context_edge_attr = batch.demonstration_context_edge_attr[demonstration_edges, :]
            # demonstration_action = batch.demonstration_action[batch.demonstration_action_batch < n_demonstrations]
            # demonstration_idx = batch.demonstration_idx[batch.demonstration_idx_batch < n_demonstrations]
            # demonstration_conv_aggr_out_idx = batch.demonstration_conv_aggr_out_idx[demonstration_edges]
            # demonstration_encoding_broadcast = batch.demonstration_encoding_broadcast[batch.demonstration_idx_batch < n_demonstrations]
            # demonstration_action_batch = batch.demonstration_action_batch[batch.demonstration_action_batch < n_demonstrations]

            demonstration_batch = DemonstrationBatch(
                batch.demonstration_context_edge_index[:, demonstration_edges],
                batch.demonstration_context_edge_attr[demonstration_edges, :],
                batch.demonstration_action[batch.demonstration_action_batch < n_demonstrations],
                batch.demonstration_idx[batch.demonstration_idx_batch < n_demonstrations],
                batch.demonstration_conv_aggr_out_idx[demonstration_edges],
                batch.demonstration_encoding_broadcast[batch.demonstration_idx_batch < n_demonstrations],
                batch.demonstration_action_batch[batch.demonstration_action_batch < n_demonstrations])
        else:
            demonstration_batch = None
        return batch, torch.from_numpy(weights), idxes, data_ids, is_demonstration, demonstration_batch

    def sgd_step(self, batch, importance_sampling_correction_weights=None, is_demonstration=None, demonstration_batch=None):
        """ implement the basic DQN optimization step """

        # # old replay buffer returned transitions as separated Transition objects
        # # todo - batch before calling sgd_step to save GPU time on distributed learner
        # batch = Transition.create_batch(transitions).to(self.device)  #, follow_batch=['x_c', 'x_v', 'x_a', 'ns_x_c', 'ns_x_v', 'ns_x_a']).to(self.device)
        batch = batch.to(self.device)
        action_batch = batch.a

        # Compute Q(s, a):
        # the model computes Q(s,:), then we select the columns of the actions taken, i.e action_batch.
        # TODO: transformer & learning from demonstrations:
        #  for each cut, we need to compute the supervising expert loss
        #  J_E = max( Q(s,a) + l(a,a_E) ) - Q(s,a_E).
        #  in order to compute the max term
        #  we need to compute the whole sequence of q_values as done in inference,
        #  and at each decoder iteration, to compute J_E.
        #  How can we do it efficiently?
        #  The cuts encoding are similar for all those cut-decisions, and can be computed only once.
        #  In order to have the full set of Q values,
        #  we can replicate the cut encodings |selected|+1 times and
        #  expand the decoder context to the full sequence of edge attributes accordingly.
        #  Each encoding replication with its corresponding context, can define a separate graph,
        #  batched together, and fed in parallel into the decoder.
        #  For the first |selected| replications we need to compute a single J_E for the corresponding
        #  selected cut, and for the last replication representing the q values of all the discarded cuts,
        #  we need to compute the |discarded| J_E losses for all the discarded cuts.
        #  Note: each one of those J_E losses should consider only the "remaining" cuts.
        policy_output = self.policy_net(
            x_c=batch.x_c,
            x_v=batch.x_v,
            x_a=batch.x_a,
            edge_index_c2v=batch.edge_index_c2v,
            edge_index_a2v=batch.edge_index_a2v,
            edge_attr_c2v=batch.edge_attr_c2v,
            edge_attr_a2v=batch.edge_attr_a2v,
            edge_index_a2a=batch.edge_index_a2a,
            edge_attr_a2a=batch.edge_attr_a2a,
            mode='batch'
        )
        q_values = policy_output['q_values'].gather(1, action_batch.unsqueeze(1))

        # demonstration loss:
        # TODO - currently implemented only for tqnet v3
        if is_demonstration.any():
            demonstration_loss = self.compute_demonstration_loss(batch, policy_output['cut_encoding'], is_demonstration, demonstration_batch, importance_sampling_correction_weights)
        else:
            demonstration_loss = 0

        # Compute the Bellman target for all next states.
        # Expected values of actions for non_terminal_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0] for DQN, or based on the policy_net
        # prediction for DDQN.
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was terminal.
        # The value of a state is computed as in BQN paper https://arxiv.org/pdf/1711.08946.pdf

        # for each graph in the next state batch, and for each cut, compute the q_values
        # using the target network.
        target_next_output = self.target_net(
            x_c=batch.ns_x_c,
            x_v=batch.ns_x_v,
            x_a=batch.ns_x_a,
            edge_index_c2v=batch.ns_edge_index_c2v,
            edge_index_a2v=batch.ns_edge_index_a2v,
            edge_attr_c2v=batch.ns_edge_attr_c2v,
            edge_attr_a2v=batch.ns_edge_attr_a2v,
            edge_index_a2a=batch.ns_edge_index_a2a,
            edge_attr_a2a=batch.ns_edge_attr_a2a,
            mode='batch'
        )
        target_next_q_values = target_next_output['q_values']
        # compute the target using either DQN or DDQN formula
        # todo: TQNet v3:
        #  compute the argmax across the "select" q values only.
        if self.use_transformer:
            assert self.tqnet_version == 'v3', 'v1 and v2 are no longer supported'
            if self.hparams.get('update_rule', 'DQN') == 'DQN':
                # y = r + gamma max_a' target_net(s', a')
                max_target_next_q_values_aggr, _ = scatter_max(target_next_q_values[:, 1],  # find max across the "select" q values only
                                                               batch.ns_x_a_batch,
                                                               # target index of each element in source
                                                               dim=0,  # scattering dimension
                                                               dim_size=self.batch_size)
            elif self.hparams.get('update_rule', 'DQN') == 'DDQN':
                # y = r + gamma target_net(s', argmax_a' policy_net(s', a'))
                policy_next_output = self.policy_net(
                    x_c=batch.ns_x_c,
                    x_v=batch.ns_x_v,
                    x_a=batch.ns_x_a,
                    edge_index_c2v=batch.ns_edge_index_c2v,
                    edge_index_a2v=batch.ns_edge_index_a2v,
                    edge_attr_c2v=batch.ns_edge_attr_c2v,
                    edge_attr_a2v=batch.ns_edge_attr_a2v,
                    edge_index_a2a=batch.ns_edge_index_a2a,
                    edge_attr_a2a=batch.ns_edge_attr_a2a,
                    mode='batch'
                )
                policy_next_q_values = policy_next_output['q_values']
                # compute the argmax over 'select' for each graph in batch
                _, argmax_policy_next_q_values = scatter_max(policy_next_q_values[:, 1], # find max across the "select" q values only
                                                             batch.ns_x_a_batch, # target index of each element in source
                                                             dim=0,  # scattering dimension
                                                             dim_size=self.batch_size)
                max_target_next_q_values_aggr = target_next_q_values[argmax_policy_next_q_values, 1].detach()

        else:
            if self.hparams.get('update_rule', 'DQN') == 'DQN':
                # y = r + gamma max_a' target_net(s', a')
                max_target_next_q_values = target_next_q_values.max(1)[0].detach()
            elif self.hparams.get('update_rule', 'DQN') == 'DDQN':
                # y = r + gamma target_net(s', argmax_a' policy_net(s', a'))
                policy_next_output = self.policy_net(
                    x_c=batch.ns_x_c,
                    x_v=batch.ns_x_v,
                    x_a=batch.ns_x_a,
                    edge_index_c2v=batch.ns_edge_index_c2v,
                    edge_index_a2v=batch.ns_edge_index_a2v,
                    edge_attr_c2v=batch.ns_edge_attr_c2v,
                    edge_attr_a2v=batch.ns_edge_attr_a2v,
                    edge_index_a2a=batch.ns_edge_index_a2a,
                    edge_attr_a2a=batch.ns_edge_attr_a2a,
                )
                policy_next_q_values = policy_next_output['q_values']
                argmax_policy_next_q_values = policy_next_q_values.max(1)[1].detach()
                max_target_next_q_values = target_next_q_values.gather(1, argmax_policy_next_q_values).detach()

            # aggregate the action-wise values using mean or max,
            # and generate for each graph in the batch a single value
            max_target_next_q_values_aggr = self.value_aggr(max_target_next_q_values,  # source vector
                                                            batch.ns_x_a_batch,  # target index of each element in source
                                                            dim=0,  # scattering dimension
                                                            dim_size=self.batch_size)   # output tensor size in dim after scattering

        # override with zeros the values of terminal states which are zero by convention
        max_target_next_q_values_aggr[batch.ns_terminal] = 0
        # broadcast the aggregated next state q_values to cut-level graph wise, to bootstrap the cut-level rewards
        target_next_q_values_broadcast = max_target_next_q_values_aggr[batch.x_a_batch]

        # now compute the cut-level target
        reward_batch = batch.r
        target_q_values = reward_batch + (self.gamma ** self.nstep_learning) * target_next_q_values_broadcast

        # Compute Huber loss
        # todo - support importance sampling correction - double check
        if self.use_per:
            # broadcast each transition importance sampling weight to all its cut-level losses
            importance_sampling_correction_weights = importance_sampling_correction_weights.to(self.device)[batch.x_a_batch]
            # multiply cut-level loss and importance sampling correction weight, and average
            n_step_loss = (importance_sampling_correction_weights * F.smooth_l1_loss(q_values, target_q_values.unsqueeze(1), reduction='none')).mean()
        else:
            # generate equal weights for all losses
            n_step_loss = F.smooth_l1_loss(q_values, target_q_values.unsqueeze(1))

        # loss = F.smooth_l1_loss(q_values, target_q_values.unsqueeze(1)) - original pytorch example
        self.n_step_loss_moving_avg = 0.95 * self.n_step_loss_moving_avg + 0.05 * n_step_loss.detach().cpu().numpy()

        # combine all losses (DQN and demonstration loss)
        loss = self.hparams.get('n_step_loss_coef', 1.0) * n_step_loss + self.hparams.get('demonstration_loss_coef', 0.5) * demonstration_loss

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.num_sgd_steps_done += 1
        # todo - for distributed learning return losses to update priorities - double check
        if self.use_per:
            # for transition, compute the norm of its TD-error
            td_error = torch.abs(q_values - target_q_values.unsqueeze(1)).detach()
            td_error = torch.clamp(td_error, min=1e-8)
            # (to compute p,q norm take power p and compute sqrt q)
            td_error_l2_norm = torch.sqrt(scatter_add(td_error ** 2, batch.x_a_batch, # target index of each element in source
                                                      dim=0,                          # scattering dimension
                                                      dim_size=self.batch_size))      # output tensor size in dim after scattering

            new_priorities = td_error_l2_norm.squeeze().cpu().numpy()

            return new_priorities
        else:
            return None

    def compute_demonstration_loss(self, batch, cut_encoding_batch, is_demonstration, demonstration_batch, importance_sampling_correction_weights):
        # todo - continue here. verify batching and everything.
        # filter non demonstration data
        n_demonstrations = sum(is_demonstration)  # number of demonstration transitions in batch
        cut_encoding_batch = cut_encoding_batch[batch.x_a_batch < n_demonstrations]
        # remove from edge_index (and corresponding edge_attr) edges which reference cuts of index greater than
        # max_demonstration_edge_index = cut_encoding_batch.shape[0] - 1
        assert demonstration_batch.context_edge_index.max() == cut_encoding_batch.shape[0] - 1
        # demonstration_edges = batch.demonstration_context_edge_index[0, :] < max_demonstration_edge_index
        # demonstration_context_edge_index = batch.demonstration_context_edge_index[:, demonstration_edges]
        # demonstration_context_edge_attr = batch.demonstration_context_edge_attr[demonstration_edges, :]
        # demonstration_action = batch.demonstration_action[batch.demonstration_action_batch < n_demonstrations]
        # demonstration_idx = batch.demonstration_idx[batch.demonstration_idx_batch < n_demonstrations]
        # demonstration_conv_aggr_out_idx = batch.demonstration_conv_aggr_out_idx[demonstration_edges]
        # demonstration_encoding_broadcast = batch.demonstration_encoding_broadcast[batch.demonstration_idx_batch < n_demonstrations]
        # demonstration_action_batch = batch.demonstration_action_batch[batch.demonstration_action_batch < n_demonstrations]
        #
        # demonstration_batch = DemonstrationBatch(
        #     demonstration_context_edge_index,
        #     demonstration_context_edge_attr,
        #     demonstration_action,
        #     demonstration_idx,
        #     demonstration_conv_aggr_out_idx,
        #     demonstration_encoding_broadcast,
        #     demonstration_action_batch)

        demonstration_context_edge_index = demonstration_batch[0].to(self.device)
        demonstration_context_edge_attr = demonstration_batch[1].to(self.device)
        demonstration_action = demonstration_batch[2].to(self.device)
        demonstration_idx = demonstration_batch[3].to(self.device)
        demonstration_conv_aggr_out_idx = demonstration_batch[4].to(self.device)
        demonstration_encoding_broadcast = demonstration_batch[5].to(self.device)
        demonstration_action_batch = demonstration_batch[6].to(self.device)

        # predict cut-level q values
        cut_decoding = self.policy_net.decoder_conv.conv_demonstration(
            x=cut_encoding_batch,
            edge_index=demonstration_context_edge_index,
            edge_attr=demonstration_context_edge_attr,
            aggr_idx=demonstration_conv_aggr_out_idx,
            encoding_broadcast=demonstration_encoding_broadcast
        )
        q_values = self.policy_net.q(cut_decoding)
        # average "discard" q_values
        discard_q_values = scatter_mean(q_values[:, 0], demonstration_idx, dim=0)
        assert discard_q_values.shape[0] == demonstration_idx[-1] + 1

        # interleave the "select" and "discard" q values
        pooled_q_values = torch.empty(size=(q_values.shape[0] + discard_q_values.shape[0],), dtype=torch.float32, device=self.device)
        pooled_q_values[torch.arange(q_values.shape[0], device=self.device) + demonstration_idx] = q_values[:, 1]  # "select" q_values
        discard_idxes = scatter_add(torch.ones_like(demonstration_idx, device=self.device), demonstration_idx, dim=0).cumsum(0) + torch.arange(discard_q_values.shape[0], device=self.device)
        pooled_q_values[discard_idxes] = discard_q_values

        # add large margin
        large_margin = torch.full_like(pooled_q_values,
                                       fill_value=self.hparams.get('demonstration_large_margin', 0.1),
                                       device=self.device)
        large_margin[demonstration_action] = 0
        q_plus_l = pooled_q_values + large_margin

        # compute demonstration loss J_E = max [Q(s,a) + large margin] - Q(s, a_E)
        q_a_E = pooled_q_values[demonstration_action]
        # todo - extend demonstration_idx with the "discard" entries, then take max
        demonstration_idx_ext = torch.empty_like(q_plus_l, dtype=torch.long, device=self.device)
        demonstration_idx_ext[torch.arange(q_values.shape[0], device=self.device) + demonstration_idx] = demonstration_idx
        demonstration_idx_ext[discard_idxes] = torch.arange(demonstration_idx[-1] + 1, device=self.device)

        max_q_plus_l, _ = scatter_max(q_plus_l, demonstration_idx_ext, dim=0)
        losses = max_q_plus_l - q_a_E

        # broadcast transition-level importance sampling correction weights to actions
        weights_broadcasted = importance_sampling_correction_weights[demonstration_action_batch].to(self.device)
        loss = (losses * weights_broadcasted).mean()

        # log demonstration loss moving average
        self.demonstration_loss_moving_avg = 0.95 * self.demonstration_loss_moving_avg + 0.05 * loss.detach().cpu().numpy()
        return loss

    # done
    def update_target(self):
        # Update the target network, copying all weights and biases in DQN
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.print(f'target net updated (sgd step = {self.num_sgd_steps_done})')

    # done
    def set_eval_mode(self):
        self.training = False
        self.policy_net.eval()

    # done
    def set_training_mode(self):
        self.training = True
        self.policy_net.train()

    # done
    def log_stats(self, global_step=None, info={}, log_directly=False):
        """
        Average tmp_stats_buffer values, log to tensorboard dir,
        and reset tmp_stats_buffer for the next round.
        This function should be called periodically during training,
        and at the end of every validation/test set evaluation
        save_best should be set to the best model according to the agent performnace on the validation set.
        If the model has shown its best so far, we save the model parameters and the dualbound/gap curves
        The global_step in plots is by default the number of policy updates.
        This global_step measure holds both for single threaded DQN and distributed DQN,
        where self.num_policy_updates is updated in
            single threaded DQN - every time optimize_model() is executed
            distributed DQN - every time the workers' policy is updated
        Tracking the global_step is essential for "resume".
        TODO adapt function to run on learner and workers separately.
            learner - plot loss
            worker - plot auc etc.
            test_worker - plot valid/test auc frac and figures
            in single thread - these are all true.
            need to separate workers' logdir in the distributed main script
        """
        # dictionary of all metrics to log
        log_dict = {}
        actor_name = '_' + self.print_prefix.replace('[', '').replace(']', '').replace(' ', '_') if len(self.print_prefix) > 0 else ''
        # actor_name = self.print_prefix.replace('[', '').replace(']', '').replace(' ', '') + '_'
        if global_step is None:
            global_step = self.num_param_updates

        print(self.print_prefix, f'Global step: {global_step} | {self.dataset_name}\t|', end='')
        cur_time_sec = time() - self.start_time + self.walltime_offset

        # log the average loss of the last training session
        print('{}-step Loss: {:.4f} | '.format(self.nstep_learning, self.n_step_loss_moving_avg), end='')
        print('Demonstration Loss: {:.4f} | '.format(self.demonstration_loss_moving_avg), end='')
        print(f'SGD Step: {self.num_sgd_steps_done} | ', end='')
        # todo wandb
        log_dict['Nstep_Loss'] = self.n_step_loss_moving_avg
        log_dict['Demonstration_Loss'] = self.demonstration_loss_moving_avg

        if log_directly:
            # todo wandb modify log dict keys with actor_name, or maybe agging is better?
            wandb.log(log_dict, step=global_step)

        # print the additional info
        for k, v in info.items():
            print(k + ': ' + v + ' | ', end='')

        # print times
        d = int(np.floor(cur_time_sec/(3600*24)))
        h = int(np.floor(cur_time_sec/3600) - 24*d)
        m = int(np.floor(cur_time_sec/60) - 60*(24*d + h))
        s = int(cur_time_sec) % 60
        print('Iteration Time: {:.1f}[sec]| '.format(cur_time_sec - self.last_time_sec), end='')
        print('Total Time: {}-{:02d}:{:02d}:{:02d}'.format(d, h, m, s))
        self.last_time_sec = cur_time_sec

        self.test_perf_list = []  # reset for the next testset

        return global_step, log_dict

    # done
    def save_checkpoint(self, filepath=None):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'num_env_steps_done': self.num_env_steps_done,
            'num_sgd_steps_done': self.num_sgd_steps_done,
            'num_param_updates': self.num_param_updates,
            'i_episode': self.i_episode,
            'walltime_offset': time() - self.start_time + self.walltime_offset,
            'best_perf': self.best_perf,
            'n_step_loss_moving_avg': self.n_step_loss_moving_avg,
        }, filepath if filepath is not None else self.checkpoint_filepath)
        if self.hparams.get('verbose', 1) > 1:
            print(self.print_prefix, 'Saved checkpoint to: ', filepath if filepath is not None else self.checkpoint_filepath)

    # done
    def _save_if_best(self):
        """Save the model if show the best performance on the validation set.
        The performance is the -(dualbound/gap auc),
        according to the DQN objective"""
        perf = -np.mean(self.tmp_stats_buffer[self.dqn_objective])
        if perf > self.best_perf[self.dataset_name]:
            self.best_perf[self.dataset_name] = perf
            self.save_checkpoint(filepath=os.path.join(self.run_dir, f'best_{self.dataset_name}_checkpoint.pt'))

    # done
    def load_checkpoint(self, filepath=None):
        if filepath is None:
            filepath = self.checkpoint_filepath
        if not os.path.exists(filepath):
            print(self.print_prefix, 'Checkpoint file does not exist! starting from scratch.')
            return
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.num_env_steps_done = checkpoint['num_env_steps_done']
        self.num_sgd_steps_done = checkpoint['num_sgd_steps_done']
        self.num_param_updates = checkpoint['num_param_updates']
        self.i_episode = checkpoint['i_episode']
        self.walltime_offset = checkpoint['walltime_offset']
        self.best_perf = checkpoint['best_perf']
        self.n_step_loss_moving_avg = checkpoint['n_step_loss_moving_avg']
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        print(self.print_prefix, 'Loaded checkpoint from: ', filepath)

    # done
    def initialize_training(self):
        # fix random seed for all experiment
        if self.hparams.get('seed', None) is not None:
            np.random.seed(self.hparams['seed'])
            torch.manual_seed(self.hparams['seed'])

        # initialize agent
        self.set_training_mode()
        if self.hparams.get('resume', False):
            self.load_checkpoint()
            # initialize prioritized replay buffer internal counters, to continue beta from the point it was
            if self.use_per:
                self.memory.num_sgd_steps_done = self.num_sgd_steps_done

    def print(self, expr):
        print(self.print_prefix, expr)

