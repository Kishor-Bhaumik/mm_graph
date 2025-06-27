import torch

import torch.nn.functional as F
import dgl
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler, as_edge_prediction_sampler, \
    negative_sampler
import tqdm

import os
import numpy as np
from sklearn import metrics
import pickle
import pdb
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
import time
import optuna
import os
import sys
import pickle

import pandas as pd
import numpy as np
import torch
import dgl

import dgl.nn as dglnn
import tqdm
import torch.nn as nn
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
from dgl.nn import GATv2Conv
from dgl.nn.pytorch.conv import GINConv
from torch.nn import Linear

class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, 'mean'))
        # n-layer GraphSAGE-mean
        for i in range(num_layers - 1):
            self.layers.append(dglnn.SAGEConv(hid_size, hid_size, 'mean'))
        self.hid_size = hid_size
        self.predictor = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1))

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        h_pos = self.predictor(h[pos_src] * h[pos_dst])
        h_neg = self.predictor(h[neg_src] * h[neg_dst])
        return h_pos, h_neg

    def inference(self, g, device, batch_size):
        """Layer-wise inference algorithm to compute GNN node embeddings."""
        feat = g.ndata['feat'].float()
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
        dataloader = DataLoader(
            g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
            batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=0)
        buffer_device = torch.device('cpu')
        pin_memory = (buffer_device != device)
        for l, layer in enumerate(self.layers):
            y = torch.empty(g.num_nodes(), self.hid_size, device=buffer_device,
                            pin_memory=pin_memory)
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader, desc='Inference'):
                x = feat[input_nodes]
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                y[output_nodes] = h.to(buffer_device)
            feat = y
        return y
    

class Logger(object):
    def __init__(self, runs):
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert 0 <= run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 0].argmax().item()
            log.info(f'Run {run + 1:02d}:')
            log.info(f'Highest Valid: {result[:, 0].max():.2f}')
            log.info(f'   Final Test: {result[argmax, 1]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                valid = r[:, 0].max().item()
                test = r[r[:, 0].argmax(), 1].item()
                best_results.append((valid, test))

            best_result = torch.tensor(best_results)

            log.info(f'All runs:')
            r = best_result[:, 0]
            log.info(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            log.info(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

log = logging.getLogger(__name__)


def to_bidirected_with_reverse_mapping(g):
    """Makes a graph bidirectional, and returns a mapping array ``mapping`` where ``mapping[i]``
    is the reverse edge of edge ID ``i``. Does not work with graphs that have self-loops.
    """
    g_simple, mapping = dgl.to_simple(
        dgl.add_reverse_edges(g), return_counts='count', writeback_mapping=True)
    c = g_simple.edata['count']
    num_edges = g.num_edges()
    mapping_offset = torch.zeros(g_simple.num_edges() + 1, dtype=g_simple.idtype)
    mapping_offset[1:] = c.cumsum(0)
    idx = mapping.argsort()
    idx_uniq = idx[mapping_offset[:-1]]
    reverse_idx = torch.where(idx_uniq >= num_edges, idx_uniq - num_edges, idx_uniq + num_edges)
    reverse_mapping = mapping[reverse_idx]
    # sanity check
    src1, dst1 = g_simple.edges()
    src2, dst2 = g_simple.find_edges(reverse_mapping)
    assert torch.equal(src1, dst2)
    assert torch.equal(src2, dst1)
    return g_simple, reverse_mapping


class LinkPredictionDataset(object):
    def __init__(self, root: str, feat_name: str, edge_split_type: str, verbose: bool=True, device: str='cpu'):
        """
        Args:
            root (str): root directory to store the dataset folder.
            feat_name (str): the name of the node features, e.g., "t5vit".
            edge_split_type (str): the type of edge split, can be "random" or "hard".
            verbose (bool): whether to print the information.
            device (str): device to use.
        """
        root = os.path.normpath(root)
        self.name = os.path.basename(root)
        self.verbose = verbose
        self.root = root
        self.feat_name = feat_name
        self.edge_split_type = edge_split_type
        self.device = device
        if self.verbose:
            print(f"Dataset name: {self.name}")
            print(f'Feature name: {self.feat_name}')
            print(f'Edge split type: {self.edge_split_type}')
            print(f'Device: {self.device}')
        
        edge_split_path = os.path.join(root, f'lp-edge-split.pt')
        self.edge_split = torch.load(edge_split_path, map_location=self.device)
        feat_path = os.path.join(root, f'{self.feat_name}_feat.pt')
        feat = torch.load(feat_path, map_location='cpu')
        self.num_nodes = feat.shape[0]
        self.graph = dgl.graph((
            self.edge_split['train']['source_node'],
            self.edge_split['train']['target_node'],
        ), num_nodes=self.num_nodes).to('cpu')
        self.graph.ndata['feat'] = feat

    def get_edge_split(self):
        return self.edge_split

    def __getitem__(self, idx: int):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph

    def __len__(self):
        return 1
    
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))

PROJETC_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(PROJETC_DIR, '../')
CONFIG_DIR = os.path.join(PROJETC_DIR, "configs")
log = logging.getLogger(__name__)

def compute_mrr_esci(
        model, 
        node_emb, 
        src, 
        dst, 
        neg_dst, 
        device, 
        batch_size=500, 
        preload_node_emb=True,
        use_concat = False, 
        use_dot = False
    ):
    """Compute Mean Reciprocal Rank (MRR) in batches in esci dataset."""

    # gpu may be out of memory for large datasets
    if preload_node_emb:
        node_emb = node_emb.to(device)

    rr = torch.zeros(src.shape[0])
    hits_at_10 = torch.zeros(src.shape[0])
    hits_at_1 = torch.zeros(src.shape[0])
    for start in tqdm.trange(0, src.shape[0], batch_size, desc='Evaluate'):
        end = min(start + batch_size, src.shape[0])
        all_dst = torch.cat([dst[start:end, None], neg_dst[start:end]], 1)
        h_src = node_emb[src[start:end]][:, None, :].to(device)
        h_dst = node_emb[all_dst.view(-1)].view(*all_dst.shape, -1).to(device)
        if use_concat:
            h_src = h_src.repeat(1, 1001, 1)
            pred = model.predictor(torch.cat((h_src, h_dst), dim=2)).squeeze(-1)
        elif use_dot:
            pred = model.decoder(h_src * h_dst).squeeze(-1)
        else:
            pred = model.predictor(h_src * h_dst).squeeze(-1)
        #import pdb; pdb.set_trace()
        y_pred_pos = pred[:, 0]
        y_pred_neg = pred[:, 1:]
        y_pred_pos = y_pred_pos.view(-1, 1)
        # optimistic rank: "how many negatives have at least the positive score?"
        # ~> the positive is ranked first among those with equal score
        optimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
        # pessimistic rank: "how many negatives have a larger score than the positive?"
        # ~> the positive is ranked last among those with equal score
        pessimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
        ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
        hits_at_10[start:end] = ranking_list<=10
        hits_at_1[start:end] = ranking_list<=1
        mrr_list = 1. / ranking_list.to(torch.float)
        rr[start:end] = mrr_list
    MRR = rr.mean()
    Hits_10 = hits_at_10.sum()/src.shape[0]
    Hits_1 = hits_at_1.sum()/src.shape[0]

    return MRR, Hits_10, Hits_1




def train(cfg, device, g, reverse_eids, seed_edges, model, edge_split, logger, run, eval_batch_size=1000):
    # create sampler & dataloader
    total_it = 1000 * 512 / cfg.batch_size
    if not os.path.exists(cfg.checkpoint_folder):
        os.makedirs(cfg.checkpoint_folder)
    checkpoint_path = cfg.checkpoint_folder + cfg.model_name + "_" + cfg.dataset + "_" + "batch_size_" + str(
        cfg.batch_size) + "_n_layers_" + str(cfg.num_layers) + "_hidden_dim_" + str(cfg.hidden_dim) + "_lr_" + str(
        cfg.lr) + "_exclude_degree_" + str(cfg.exclude_target_degree) + "_full_neighbor_" + str(
        cfg.full_neighbor) + "_accu_num_" + str(cfg.accum_iter_number) + "_trail_" + str(run) + "_best.pth"
    if cfg.full_neighbor:
        log.info("We use the full neighbor of the target node to train the models. ")
        sampler = MultiLayerFullNeighborSampler(num_layers=cfg.num_layers, prefetch_node_feats=['feat'])
    else:
        log.info("We sample the neighbor node of the target node to train the models. ")
        sampler = NeighborSampler([cfg.num_of_neighbors] * cfg.num_layers, prefetch_node_feats=['feat'])
    log.info("We exclude the training target. ")
    sampler = as_edge_prediction_sampler(
        sampler, exclude="reverse_id", reverse_eids=reverse_eids, negative_sampler=negative_sampler.Uniform(1))
    use_uva = (cfg.mode == 'mixed')
    dataloader = DataLoader(
        g, seed_edges, sampler,
        device=device, batch_size=cfg.batch_size, shuffle=True,
        drop_last=False, num_workers=0, use_uva=use_uva)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        opt, 
        step_size=cfg.lr_scheduler_step_size,
        gamma=cfg.lr_scheduler_gamma,
    )
    optuna_acc = 0
    for epoch in range(cfg.n_epochs):
        model.train()
        total_loss = 0
        # batch accumulation parameter
        accum_iter = cfg.accum_iter_number

        log.info('Training...')
        for it, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(dataloader):
            #import pdb; pdb.set_trace()
            # pair_graph: all positive edge pairs in this batch, stored  as a graph
            # neg_pair_graph: all negative edge pairs in this batch, stored as a graph
            # blocks: each block is the aggregated graph as input for each layer
            x = blocks[0].srcdata['feat'].float()
            pos_score, neg_score = model(pair_graph, neg_pair_graph, blocks, x)
            score = torch.cat([pos_score, neg_score])
            pos_label = torch.ones_like(pos_score)
            neg_label = torch.zeros_like(neg_score)
            labels = torch.cat([pos_label, neg_label])
            loss = F.binary_cross_entropy_with_logits(score, labels)
            (loss / accum_iter).backward()
            if ((it + 1) % accum_iter == 0) or (it + 1 == len(dataloader)) or (it + 1 == total_it):
                # Update Optimizer
                opt.step()
                opt.zero_grad()
            total_loss += loss.item()
            if (it + 1) == total_it:
                break

        lr_scheduler.step()

        log.info("Epoch {:05d} | Loss {:.4f}".format(epoch, total_loss / (it + 1)))
        if (epoch + 1) % cfg.log_steps == 0:
            model.eval()
            log.info('Validation/Testing...')
            with torch.no_grad():
                node_emb = model.inference(g, device, eval_batch_size)
                results = []

                log.info("do evaluation on training examples: check if can be overfitted")
                torch.manual_seed(12345)
                num_sampled_nodes = edge_split['valid']['target_node_neg'].size(dim=0)
                idx = torch.randperm(edge_split['train']['source_node'].numel())[:num_sampled_nodes]
                edge_split['eval_train'] = {
                    'source_node': edge_split['train']['source_node'][idx],
                    'target_node': edge_split['train']['target_node'][idx],
                    'target_node_neg': edge_split['valid']['target_node_neg'],
                }

                src = edge_split['eval_train']['source_node'].to(node_emb.device)
                dst = edge_split['eval_train']['target_node'].to(node_emb.device)
                neg_dst = edge_split['eval_train']['target_node_neg'].to(node_emb.device)

                use_concat = cfg.use_concat
                use_dot = cfg.model_name == "Dot"
                mrr, hits_at_10, hits_at_1 = compute_mrr_esci(model, node_emb, src, dst, neg_dst, device, preload_node_emb=cfg.preload_node_emb, use_concat=use_concat, use_dot=use_dot)
                log.info('Train MRR {:.4f} '.format(mrr.item()))
                if cfg.no_eval is False:
                    valid_mrr = []
                    valid_h_10 = []
                    valid_h_1 = []
                    test_mrr = []
                    test_h_10 = []
                    test_h_1 = []
                    for split in ['valid', 'test']:
                        if cfg.dataset == "ogbl-citation2":
                            evaluator = Evaluator(name=cfg.dataset)
                            src = edge_split[split]['source_node'].to(node_emb.device)
                            dst = edge_split[split]['target_node'].to(node_emb.device)
                            neg_dst = edge_split[split]['target_node_neg'].to(node_emb.device)
                            results.append(compute_mrr(model, evaluator, node_emb, src, dst, neg_dst, device))
                        else:
                            src = edge_split[split]['source_node'].to(node_emb.device)
                            dst = edge_split[split]['target_node'].to(node_emb.device)
                            neg_dst = edge_split[split]['target_node_neg'].to(node_emb.device)
                            results.append(
                                compute_mrr_esci(model, node_emb, src, dst, neg_dst, device, preload_node_emb=cfg.preload_node_emb, use_concat=use_concat, use_dot=use_dot)
                            )
                    valid_mrr.append(results[0][0].item())
                    valid_h_10.append(results[0][1].item())
                    valid_h_1.append(results[0][2].item())
                    test_mrr.append(results[1][0].item())
                    test_h_10.append(results[1][1].item())
                    test_h_1.append(results[1][2].item())

                    # save best checkpoint
                    valid_result, test_result = results[0][0].item(), results[1][0].item()
                    
                    # we want to find the best previous checkpoint
                    # if there is no previous checkpoint, set it to 0
                    # Warning: it only works for MRR and Hit@N.
                    if len(logger.results[run]) > 0:
                        previous_best_valid_result = torch.tensor(logger.results[run])[:, 0].max().item()
                    else:  # length = 0
                        previous_best_valid_result = 0.0
                    
                    if valid_result > previous_best_valid_result:
                        log.info("Saving checkpoint. ")
                        torch.save({
                            'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                        }, checkpoint_path)
                        optuna_acc = test_result

                    logger.add_result(run, [valid_result, test_result])
                    log.info('Validation MRR {:.4f}, Test MRR {:.4f}'.format(valid_result, test_result))
                    log.info('Validation Hits@10 {:.4f}, Test Hits@10 {:.4f}'.format(results[0][1].item(), results[1][1].item()))
                    log.info('Validation Hits@1 {:.4f}, Test Hits@1 {:.4f}'.format(results[0][2].item(), results[1][2].item()))
    logger.print_statistics(run)
    return results[1][0].item(), results[1][1].item(), results[1][2].item()


@hydra.main(config_path=CONFIG_DIR, config_name="defaults", version_base='1.2')
def main(cfg: DictConfig):
    log.info('Loading data')
    data_path = '/home/kbhau001/llm/mm-graph-benchmark/Multimodal-Graph-Completed-Graph/' # replace this with the path where you save the datasets
    dataset_name = 'books-lp'
    feat_name = 't5vit'
    edge_split_type = 'hard'
    verbose = True
    device = ('cuda' if cfg.mode == 'puregpu' else 'cpu') # use 'cuda' if GPU is available

    dataset = LinkPredictionDataset(
        root=os.path.join(data_path, dataset_name),
        feat_name=feat_name,
        edge_split_type=edge_split_type,
        verbose=verbose,
        device=device
    )

    g = dataset.graph
    # type(graph) would be dgl.DGLGraph
    # use graph.ndata['feat'] to get the features

    edge_split = dataset.get_edge_split()
    g = dgl.remove_self_loop(g)
    log.info("remove isolated nodes")
    g, reverse_eids = to_bidirected_with_reverse_mapping(g)
    g = g.to('cuda' if cfg.mode == 'puregpu' else 'cpu')
    num_nodes = g.number_of_nodes()
    reverse_eids = reverse_eids.to(device)
    seed_edges = torch.arange(g.num_edges()).to(device)
    
    in_size = g.ndata['feat'].shape[1]
    logger = Logger(cfg.runs)

    mrrs = []
    h1s = []
    h10s = []
    
    for run in range(cfg.runs):
        log.info("Run {}/{}".format(run + 1, cfg.runs))
        if cfg.model_name == "SAGE":
            model = SAGE(in_size, cfg.hidden_dim, cfg.num_layers).to(device)
        else:
            raise ValueError(f"Model '{cfg.model_name}' is not supported")
        # model training
        log.info('Training...')
        # log.info(edge_split['test'].keys())
        mrr, h10, h1 = train(cfg, device, g, reverse_eids, seed_edges, model, edge_split, logger, run)
        mrrs.append(mrr)
        h10s.append(h10)
        h1s.append(h1)
    logger.print_statistics()
    mean_mrr = torch.mean(torch.tensor(mrrs)).item()
    std_mrr = torch.std(torch.tensor(mrrs)).item()
    mean_h1 = torch.mean(torch.tensor(h1s)).item()
    std_h1 = torch.std(torch.tensor(h1s)).item()
    mean_h10 = torch.mean(torch.tensor(h10s)).item()
    std_h10 = torch.std(torch.tensor(h10s)).item()
    print(mean_mrr)
    print(std_mrr)
    print(mean_h1)
    print(std_h1)
    print(mean_h10)
    print(std_h10)    
    return mean_mrr


if __name__=='__main__':
    main()