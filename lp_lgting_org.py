import torch
import torch.nn.functional as F
import dgl
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler, as_edge_prediction_sampler, \
    negative_sampler
import tqdm
import time
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

# PyTorch Lightning imports
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT

import warnings
# remove all warnings
warnings.filterwarnings("ignore")


"""
Do you understand the code ? answer very shortly

converted this code to pytorch lighting? is the conversion okay? or does it have any mistake for which the result could show difference .
"""
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


class GraphLinkPredictionDataModule(pl.LightningDataModule):
    def __init__(self, cfg, graph, edge_split, reverse_eids, seed_edges, device):
        super().__init__()
        self.cfg = cfg
        self.graph = graph
        self.edge_split = edge_split
        self.reverse_eids = reverse_eids
        self.seed_edges = seed_edges
        self.device_str = device
        
    def setup(self, stage=None):
        # Create sampler
        if self.cfg.full_neighbor:
            sampler = MultiLayerFullNeighborSampler(num_layers=self.cfg.num_layers, prefetch_node_feats=['feat'])
        else:
            sampler = NeighborSampler([self.cfg.num_of_neighbors] * self.cfg.num_layers, prefetch_node_feats=['feat'])
        
        self.sampler = as_edge_prediction_sampler(
            sampler, exclude="reverse_id", reverse_eids=self.reverse_eids, 
            negative_sampler=negative_sampler.Uniform(1))
        
    def train_dataloader(self):
        use_uva = (self.cfg.mode == 'mixed')
        return DataLoader(
            self.graph, self.seed_edges, self.sampler,
            device=self.device_str, batch_size=self.cfg.batch_size, shuffle=True,
            drop_last=False, num_workers=0, use_uva=use_uva)


    # --- ADD THESE METHODS ---
    def val_dataloader(self):
        # The validation step is a no-op, and evaluation happens on the full graph.
        # Returning the train_dataloader is acceptable here to trigger the validation hooks.
        use_uva = (self.cfg.mode == 'mixed')
        return DataLoader(
            self.graph, self.seed_edges, self.sampler,
            device=self.device_str, batch_size=self.cfg.batch_size, shuffle=False,
            drop_last=False, num_workers=0, use_uva=use_uva)

    def test_dataloader(self):
        use_uva = (self.cfg.mode == 'mixed')
        return DataLoader(
            self.graph, self.seed_edges, self.sampler,
            device=self.device_str, batch_size=self.cfg.batch_size, shuffle=False,
            drop_last=False, num_workers=0, use_uva=use_uva)
    

class GraphLinkPredictionModule(pl.LightningModule):
    def __init__(self, cfg, in_size, graph, edge_split,total_it, eval_batch_size=1000):
        super().__init__()
        self.cfg = cfg
        self.total_it= total_it
        self.should_evaluate = False
        self.save_hyperparameters(ignore=['graph', 'edge_split'])

        if cfg.model_name == "SAGE":
            self.model = SAGE(in_size, cfg.hidden_dim, cfg.num_layers)
        else:
            raise ValueError(f"Model '{cfg.model_name}' is not supported")
        
        self.graph = graph
        self.edge_split = edge_split
        self.eval_batch_size = eval_batch_size
        
        # Track best validation result
        self.best_valid_result = 0.0
        self.best_test_result = 0.0
        
        # Configure automatic optimization
        self.automatic_optimization = False
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=self.cfg.lr_scheduler_step_size,
            gamma=self.cfg.lr_scheduler_gamma,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
            },
        }
    
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        
        input_nodes, pair_graph, neg_pair_graph, blocks = batch
        x = blocks[0].srcdata['feat'].float()
        
        pos_score, neg_score = self.model(pair_graph, neg_pair_graph, blocks, x)
        score = torch.cat([pos_score, neg_score])
        pos_label = torch.ones_like(pos_score)
        neg_label = torch.zeros_like(neg_score)
        labels = torch.cat([pos_label, neg_label])
        loss = F.binary_cross_entropy_with_logits(score, labels)
        
        # Manual backward pass with accumulation
        accum_iter = self.cfg.accum_iter_number
        (loss / accum_iter).backward()
        

        # Get the actual number of batches in this epoch
        num_batches = self.trainer.num_training_batches
        if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == num_batches):
            opt.step()
            opt.zero_grad()
                            
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.cfg.batch_size)
        self.log('learning_rate', sch.get_last_lr()[0], on_step=True, on_epoch=True, batch_size=self.cfg.batch_size)


        
        return loss
    
    def on_train_epoch_end(self):
        sch = self.lr_schedulers()
        sch.step()
    
        if (self.current_epoch + 1) % self.cfg.log_steps == 0 or self.should_evaluate:
            self._evaluate_model()
    
    def _evaluate_model(self):
          
        """Perform full model evaluation"""
        self.model.eval()
        with torch.no_grad():
            node_emb = self.model.inference(self.graph, self.device, self.eval_batch_size)
            torch.manual_seed(12345)
            num_sampled_nodes = self.edge_split['valid']['target_node_neg'].size(dim=0)
            idx = torch.randperm(self.edge_split['train']['source_node'].numel())[:num_sampled_nodes]
            eval_train_split = {
                'source_node': self.edge_split['train']['source_node'][idx],
                'target_node': self.edge_split['train']['target_node'][idx],
                'target_node_neg': self.edge_split['valid']['target_node_neg'],
            }
            
            src = eval_train_split['source_node'].to(node_emb.device)
            dst = eval_train_split['target_node'].to(node_emb.device)
            neg_dst = eval_train_split['target_node_neg'].to(node_emb.device)
            
            use_concat = self.cfg.use_concat
            use_dot = self.cfg.model_name == "Dot"
            
            train_mrr, train_hits_at_10, train_hits_at_1 = compute_mrr_esci(
                self.model, node_emb, src, dst, neg_dst, self.device, 
                preload_node_emb=self.cfg.preload_node_emb, use_concat=use_concat, use_dot=use_dot
            )
            
            self.log('train_eval_mrr', train_mrr.item(), on_epoch=True, prog_bar=True, batch_size=1)
            self.log('train_eval_hits@10', train_hits_at_10.item(), on_epoch=True, batch_size=1)
            self.log('train_eval_hits@1', train_hits_at_1.item(), on_epoch=True, batch_size=1)
            
            if not self.cfg.no_eval:
                # Evaluate on validation and test sets
                valid_results = []
                test_results = []
                
                for split in ['valid', 'test']:
                    src = self.edge_split[split]['source_node'].to(node_emb.device)
                    dst = self.edge_split[split]['target_node'].to(node_emb.device)
                    neg_dst = self.edge_split[split]['target_node_neg'].to(node_emb.device)
                    
                    mrr, hits_at_10, hits_at_1 = compute_mrr_esci(
                        self.model, node_emb, src, dst, neg_dst, self.device, 
                        preload_node_emb=self.cfg.preload_node_emb, use_concat=use_concat, use_dot=use_dot
                    )
                    
                    if split == 'valid':
                        valid_results = [mrr, hits_at_10, hits_at_1]
                    else:
                        test_results = [mrr, hits_at_10, hits_at_1]
                
                valid_result = valid_results[0].item()
                test_result = test_results[0].item()
                
                # Log validation and test metrics
                self.log('valid_mrr', valid_result, on_epoch=True, prog_bar=True)
                self.log('valid_hits@10', valid_results[1].item(), on_epoch=True)
                self.log('valid_hits@1', valid_results[2].item(), on_epoch=True)
                self.log('test_mrr', test_result, on_epoch=True, prog_bar=True)
                self.log('test_hits@10', test_results[1].item(), on_epoch=True)
                self.log('test_hits@1', test_results[2].item(), on_epoch=True)
                
                # Update best results
                if valid_result > self.best_valid_result:
                    self.best_valid_result = valid_result
                    self.best_test_result = test_result
                    
                    self.log('best_valid_mrr', self.best_valid_result, on_epoch=True)
                    self.log('best_test_mrr', self.best_test_result, on_epoch=True)
                    self.log('best_epoch', self.current_epoch, on_epoch=True)
    



PROJETC_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(PROJETC_DIR, '../')
CONFIG_DIR = os.path.join(PROJETC_DIR, "configs")
log = logging.getLogger(__name__)


@hydra.main(config_path=CONFIG_DIR, config_name="defaults", version_base='1.2')
def main(cfg: DictConfig):

    log.info('Loading data')
    data_path = '/home/kbhau001/llm/mm-graph-benchmark/Multimodal-Graph-Completed-Graph/'
    dataset_name = 'books-lp'
    feat_name = 't5vit'
    edge_split_type = 'hard'
    verbose = True
    device = ('cuda:'+str(cfg.gpu_id) if cfg.mode == 'puregpu' else 'cpu')

    dataset = LinkPredictionDataset(
        root=os.path.join(data_path, dataset_name),
        feat_name=feat_name,
        edge_split_type=edge_split_type,
        verbose=verbose,
        device=device
    )

    g = dataset.graph
    edge_split = dataset.get_edge_split()
    g = dgl.remove_self_loop(g)
    log.info("remove isolated nodes")
    g, reverse_eids = to_bidirected_with_reverse_mapping(g)
    g = g.to('cuda:'+str(cfg.gpu_id) if cfg.mode == 'puregpu' else 'cpu')
    num_nodes = g.number_of_nodes()
    reverse_eids = reverse_eids.to(device)
    seed_edges = torch.arange(g.num_edges()).to(device)
    in_size = g.ndata['feat'].shape[1]
    
    # Prepare config for logging
    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    wandb_config.update({
        'dataset_name': dataset_name,
        'feat_name': feat_name,
        'edge_split_type': edge_split_type,
        'num_nodes': num_nodes,
        'num_edges': g.num_edges(),
        'in_size': in_size
    })
    
    # Set up callbacks
    callbacks = []
    
    # Model checkpoint callback
    if not os.path.exists(cfg.checkpoint_folder):
        os.makedirs(cfg.checkpoint_folder)
    
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=cfg.checkpoint_folder,
    #     filename=f"{cfg.model_name}_{dataset_name}_batch_size_{cfg.batch_size}_n_layers_{cfg.num_layers}_hidden_dim_{cfg.hidden_dim}_lr_{cfg.lr}_exclude_degree_{cfg.exclude_target_degree}_full_neighbor_{cfg.full_neighbor}_accu_num_{cfg.accum_iter_number}_best",
    #     monitor='valid_mrr',
    #     mode='max',
    #     save_top_k=1,
    #     save_last=True,
    #     verbose=True,
    # )
    # callbacks.append(checkpoint_callback)
    
    # Set up logger
    logger = None
    if cfg.get('use_logger', True):  # Default to True if not specified
        experiment_name = "lightning_base_new" #f"{cfg.model_name}_{dataset_name}_{feat_name}"
        logger = WandbLogger(
            project="graph-link-prediction",
            name=experiment_name,
            config=wandb_config,
            tags=["link-prediction", "graph-neural-network", "pytorch-lightning"]
        )
    
    # Results storage
    all_results = []
    
    for run in range(cfg.runs):
        log.info(f"Run {run + 1}/{cfg.runs}")
        
        # Update logger for this run if using logger
        if logger is not None:
            logger.experiment.name = f"{experiment_name}_run_{run}"
  
        data_module = GraphLinkPredictionDataModule(cfg, g, edge_split, reverse_eids, seed_edges, device)
        total_it = int(1000 * 512 / cfg.batch_size)
        model = GraphLinkPredictionModule(cfg, in_size, g, edge_split,total_it, eval_batch_size=1000)
        
        trainer = Trainer(
            max_epochs=cfg.n_epochs,        
            limit_train_batches=total_it,
            check_val_every_n_epoch=None,  
            num_sanity_val_steps=0,        
            callbacks=callbacks,
            logger=logger,
            accelerator='gpu',
            devices=[cfg.gpu_id],
            log_every_n_steps=50,
            enable_checkpointing=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            precision='32',
        )
        
        # Train model
        trainer.fit(model, data_module)
        
        # Test model
        #test_results = trainer.test(model, data_module)
        
        # Store results
        run_result = {
            'run': run,
            'best_valid_mrr': model.best_valid_result,
            'best_test_mrr': model.best_test_result,
        }
        all_results.append(run_result)
        
        log.info(f"Run {run + 1} - Valid MRR: {model.best_valid_result:.4f}, Test MRR: {model.best_test_result:.4f}")
    
    # Calculate summary statistics
    valid_mrrs = [r['best_valid_mrr'] for r in all_results]
    test_mrrs = [r['best_test_mrr'] for r in all_results]
    
    mean_valid_mrr = np.mean(valid_mrrs)
    std_valid_mrr = np.std(valid_mrrs)
    mean_test_mrr = np.mean(test_mrrs)
    std_test_mrr = np.std(test_mrrs)
    
    print(f"\nFinal Results across {cfg.runs} runs:")
    print(f"Valid MRR: {mean_valid_mrr:.4f} ± {std_valid_mrr:.4f}")
    print(f"Test MRR: {mean_test_mrr:.4f} ± {std_test_mrr:.4f}")
    
    # Log summary if using logger
    if logger is not None:
        logger.log_metrics({
            'summary/valid_mrr_mean': mean_valid_mrr,
            'summary/valid_mrr_std': std_valid_mrr,
            'summary/test_mrr_mean': mean_test_mrr,
            'summary/test_mrr_std': std_test_mrr,
            'summary/num_runs': cfg.runs
        })
        
        # Log individual run results
        for i, result in enumerate(all_results):
            logger.log_metrics({
                f'runs/run_{i}_valid_mrr': result['best_valid_mrr'],
                f'runs/run_{i}_test_mrr': result['best_test_mrr']
            })
    
    return mean_test_mrr


if __name__ == '__main__':
    # time 
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    # in hours
    print(f"Total time: {(end_time - start_time) / 3600:.2f} hours")
    # in minutes
    print(f"Total time: {(end_time - start_time) / 60:.2f} minutes")
    # in seconds