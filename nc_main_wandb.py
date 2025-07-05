import time
start_time = time.time()

import argparse
import os
import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,)

from dgl.nn import GATv2Conv
import hydra
import logging
from omegaconf import DictConfig, OmegaConf

# Add wandb import
import wandb

PROJETC_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), './')
CONFIG_DIR = os.path.join(PROJETC_DIR, "configs")
log = logging.getLogger(__name__)


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers=3, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(dglnn.SAGEConv(in_size, out_size, 'mean'))
        elif num_layers == 2:
            self.layers.append(dglnn.SAGEConv(in_size, hid_size, 'mean'))
            self.layers.append(dglnn.SAGEConv(hid_size, out_size, 'mean'))
        elif num_layers == 3:
            self.layers.append(dglnn.SAGEConv(in_size, hid_size, 'mean'))
            self.layers.append(dglnn.SAGEConv(hid_size, hid_size, 'mean'))
            self.layers.append(dglnn.SAGEConv(hid_size, out_size, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l != len(self.layers) - 1 else self.out_size,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y


class NodeClassificationDataset(object):
    def __init__(self, root: str, feat_name: str, verbose: bool=True, device: str="cpu"):
        """
        Args:
            root (str): root directory to store the dataset folder.
            feat_name (str): the name of the node features, e.g., "t5vit".
            verbose (bool): whether to print the information.
            device (str): device to use.
        """
        root = os.path.normpath(root)
        self.name = os.path.basename(root)
        self.verbose = verbose
        self.root = root
        self.feat_name = feat_name
        self.device = device
        if self.verbose:
            print(f"Dataset name: {self.name}")
            print(f'Feature name: {self.feat_name}')
            print(f'Device: {self.device}')
        
        edge_path = os.path.join(root, 'nc_edges-nodeid.pt')
        self.edge = torch.tensor(torch.load(edge_path, weights_only=True), dtype=torch.int64).to(self.device)
        feat_path = os.path.join(root, f'{self.feat_name}_feat.pt')
        feat = torch.load(feat_path, map_location=self.device)
        self.num_nodes = feat.shape[0]
        
        src, dst = self.edge.t()[0], self.edge.t()[1]
        self.graph = dgl.graph((src, dst), num_nodes=self.num_nodes).to(self.device)
        self.graph.ndata['feat'] = feat
        
        labels_path = os.path.join(root, 'labels-w-missing.pt')
        self.labels = torch.tensor(torch.load(labels_path), dtype=torch.int64).to(self.device)
        self.graph.ndata['label'] = self.labels
        
        node_split_path = os.path.join(root, 'split.pt')
        self.node_split = torch.load(node_split_path)
        
        train_mask = torch.zeros(self.num_nodes, dtype=torch.bool).to(self.device)
        val_mask = torch.zeros(self.num_nodes, dtype=torch.bool).to(self.device)
        test_mask = torch.zeros(self.num_nodes, dtype=torch.bool).to(self.device)

        train_mask[self.node_split['train_idx']] = True
        val_mask[self.node_split['val_idx']] = True
        test_mask[self.node_split['test_idx']] = True

        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask
        
    def get_idx_split(self):
        return self.node_split
    
    def __getitem__(self, idx: int):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph
    
    def __len__(self):
        return 1
    
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


# Remove the old Logger class and replace with wandb logging utilities
class WandbLogger:
    def __init__(self, project_name="graph-test", run_name=None, config=None, use_logger=True):
        """Initialize wandb logger"""
        self.project_name = project_name
        self.run_name = run_name
        self.config = config
        self.run_results = []
        self.use_logger = use_logger
        
    def init_run(self, run_id, config=None):
        """Initialize a new wandb run"""
        if not self.use_logger:
            return
            
        run_name = f"{self.run_name}_run_{run_id}" if self.run_name else f"run_{run_id}"
        wandb.init(
            project=self.project_name,
            name=run_name,
            config=config or self.config,
            reinit=True,
            tags=["node-classification", "graph-neural-network"]
        )
        
    def log_metrics(self, metrics_dict, step=None):
        """Log metrics to wandb"""
        if not self.use_logger:
            return
        wandb.log(metrics_dict, step=step)
        
    def log_run_result(self, run_id, accuracy):
        """Log results for a specific run"""
        self.run_results.append({
            'run_id': run_id,
            'accuracy': accuracy
        })
        
        if not self.use_logger:
            return
            
        # Log to current wandb run
        wandb.log({
            'final_accuracy': accuracy,
            'run_id': run_id
        })
        
    def log_summary_statistics(self):
        """Log summary statistics across all runs"""
        if not self.run_results:
            return
            
        accuracies = [r['accuracy'] for r in self.run_results]
        
        acc_tensor = torch.tensor(accuracies)
        
        summary_stats = {
            'summary/accuracy_mean': acc_tensor.mean().item(),
            'summary/accuracy_std': acc_tensor.std().item(),
            'summary/num_runs': len(self.run_results)
        }
        
        if self.use_logger:
            # Log summary to a separate summary run
            wandb.init(
                project=self.project_name,
                name=f"{self.run_name}_summary" if self.run_name else "experiment_summary",
                config=self.config,
                reinit=True,
                tags=["summary", "node-classification"]
            )
            wandb.log(summary_stats)
            
            # Also log individual run results
            for i, result in enumerate(self.run_results):
                wandb.log({
                    f'runs/run_{i}_accuracy': result['accuracy']
                })
        
        log.info(f'Summary Statistics:')
        log.info(f'Accuracy: {acc_tensor.mean():.4f} ± {acc_tensor.std():.4f}')
        
        if self.use_logger:
            wandb.finish()
        
    def finish_run(self):
        """Finish current wandb run"""
        if not self.use_logger:
            return
        wandb.finish()


def to_bidirected_with_reverse_mapping(g):
    """Makes a graph bidirectional, and returns a mapping array ``mapping`` where ``mapping[i]``
    is the reverse edge of edge ID ``i``. Does not work with graphs that have self-loops.
    """
    # Store the original device
    original_device = g.device
    
    # Move graph to CPU for dgl.to_simple() operation
    g_cpu = g.to('cpu')
    
    g_simple, mapping = dgl.to_simple(
        dgl.add_reverse_edges(g_cpu), return_counts='count', writeback_mapping=True)
    c = g_simple.edata['count']
    num_edges = g_cpu.num_edges()
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
    
    # Move the graph back to the original device
    g_simple = g_simple.to(original_device)
    reverse_mapping = reverse_mapping.to(original_device)
    
    return g_simple, reverse_mapping

def evaluate(cfg, model, graph, dataloader, num_classes):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            if cfg.model_name == "MLP":
                x = blocks[-1].dstdata["feat"]
            else:
                x = blocks[0].srcdata["feat"]
            ys.append(blocks[-1].dstdata["label"])
            y_hats.append(model(blocks, x))
    pred = torch.argmax(torch.cat(y_hats), dim=1)
    acc = (pred == torch.cat(ys)).sum()/torch.cat(ys).shape[0]
    return acc



def layerwise_infer(device, graph, nid, model, num_classes, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(graph, device, batch_size) 
        nid_same_device = nid.to(pred.device)
        pred = pred[nid_same_device]
        label = graph.ndata["label"][nid].to(pred.device)
        pred = torch.argmax(pred, dim=1)
        acc = (pred == label).sum() / label.shape[0]
        
        return acc

def train(cfg, device, g, dataset, model, num_classes, run, wandb_logger):
    # create sampler & dataloader
    if not os.path.exists(cfg.checkpoint_folder):
        os.makedirs(cfg.checkpoint_folder)
    checkpoint_path = cfg.checkpoint_folder + cfg.model_name + "_" + cfg.dataset + "_" + "batch_size_" + str(
            cfg.batch_size) + "_n_layers_" + str(cfg.num_layers) + "_hidden_dim_" + str(cfg.hidden_dim) + "_lr_" + str(
            cfg.lr) + "_exclude_degree_" + str(cfg.exclude_target_degree) + "_full_neighbor_" + str(
            cfg.full_neighbor) + "_accu_num_" + str(cfg.accum_iter_number) + "_trail_" + str(run) + "_best.pth"
    
    # Fix: Handle tensor conversion properly
    train_idx = dataset['train_idx'].squeeze().to(device)  # Remove extra dimensions and ensure on correct device
    val_idx = dataset['val_idx'].squeeze().to(device)      # Remove extra dimensions and ensure on correct device
    
    if cfg.full_neighbor:
        log.info("We use the full neighbor of the target node to train the models. ")
        sampler = MultiLayerFullNeighborSampler(num_layers=cfg.num_layers, prefetch_node_feats=['feat'])
    else:
        log.info("We sample the neighbor node of the target node to train the models. ")
        sampler = NeighborSampler([cfg.num_of_neighbors] * cfg.num_layers, prefetch_node_feats=['feat'])
    use_uva = cfg.mode == "mixed"
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=512,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        device=device,
        batch_size=512,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        opt, 
        step_size=cfg.lr_scheduler_step_size,
        gamma=cfg.lr_scheduler_gamma,
    )

    best_acc = 0
    best_test_acc = 0
    for epoch in range(cfg.n_epochs):
        model.train()
        total_loss = 0
        valid_result = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            if cfg.model_name == "MLP":
                x = blocks[-1].dstdata["feat"]
            else:
                x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        acc = evaluate(cfg, model, g, val_dataloader, num_classes)
        
        # Log training metrics
        avg_loss = total_loss / (it + 1)
        current_lr = lr_scheduler.get_last_lr()[0]
        
        if cfg.use_logger:
            wandb_logger.log_metrics({
                'epoch': epoch,
                'train_loss': avg_loss,
                'val_accuracy': acc.item(),
                'learning_rate': current_lr,
                'run_id': run
            }, step=epoch)
        
        log.info(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, avg_loss, acc.item()
            )
        )
        lr_scheduler.step()

        if acc > valid_result:
            valid_result = acc
            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(),}, checkpoint_path)
            
            # test the model
            log.info("Validation...")
            log.info("Validation Accuracy {:.4f}".format(acc.item()))
            log.info("Testing...")
            test_acc = layerwise_infer(
                device, g, dataset['test_idx'].squeeze().to(device), model, num_classes, batch_size=1024
            )
            log.info("Test Accuracy {:.4f}".format(test_acc.item()))
            
            # Log test metrics
            if cfg.use_logger:
                wandb_logger.log_metrics({
                    'test_accuracy': test_acc.item(),
                    'best_val_accuracy': valid_result.item(),
                    'best_epoch': epoch,
                    'run_id': run
                }, step=epoch)
            
            best_test_acc = test_acc.item()
        
        if acc.item() > best_acc:
            best_acc = acc.item()
    
    return best_acc


@hydra.main(config_path=CONFIG_DIR, config_name="defaults", version_base='1.2')
def main(cfg: DictConfig):
    # save configs
    if not os.path.isfile("config.yaml"):
        OmegaConf.save(config=cfg, f=os.path.join("config.yaml"))

    # load and preprocess dataset
    device = torch.device('cuda:'+str(cfg.gpu_id) )

    # load and preprocess dataset
    log.info("Loading data")

    data_path = '/home/kbhau001/llm/mm-graph-benchmark/Multimodal-Graph-Completed-Graph' # replace this with the path where you save the datasets
    dataset_name = 'books-nc'
    feat_name = 't5vit'
    verbose = True

    dataset = NodeClassificationDataset(
        root=os.path.join(data_path, dataset_name),
        feat_name=feat_name,
        verbose=verbose,
        device=device
    )

    g = dataset.graph
    labels = dataset.labels
    BOOKS_PATH = os.path.join(data_path, dataset_name)
    #ndata = dataset.get_idx_split()
    if cfg.feat == 'clip':
        clip_feat = torch.load(os.path.join(BOOKS_PATH, 'clip_feat.pt'))
    elif cfg.feat == 'imagebind':
        clip_feat = torch.load(os.path.join(BOOKS_PATH, 'imagebind_feat.pt'))
    elif cfg.feat == 'dino':
        clip_feat = torch.load(os.path.join(BOOKS_PATH, 't5dino_feat.pt'))
    else: 
        clip_feat = torch.load(os.path.join(BOOKS_PATH, 't5vit_feat.pt'))
    if cfg.use_feature == 'text': 
        clip_feat = clip_feat[:,:768]    
    g.ndata['feat'] = clip_feat.to(device)
    g.ndata['label'] = labels.to(device)
    g = dgl.remove_self_loop(g)
    g, reverse_eids = to_bidirected_with_reverse_mapping(g)
    g = g.to(device)
    if cfg.dataset == 'books':
        print("using books")
        num_classes = len(torch.unique(labels))
    else:
        num_classes = 12

    splits ={}         
    splits['train_idx'] = g.ndata['train_mask'].nonzero()
    splits['val_idx'] = g.ndata['val_mask'].nonzero()
    splits['test_idx'] = g.ndata['test_mask'].nonzero()
    
    num_nodes = g.num_nodes()
    # create GraphSAGE model
    in_size = g.ndata["feat"].shape[1]
    out_size = num_classes
    
    # Initialize wandb logger
    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    wandb_config.update({
        'dataset_name': dataset_name,
        'feat_name': feat_name,
        'num_nodes': num_nodes,
        'num_edges': g.num_edges(),
        'num_classes': num_classes,
        'in_size': in_size
    })
    
    experiment_name = f"{cfg.model_name}_{dataset_name}_{feat_name}"
    wandb_logger = WandbLogger(
        project_name=cfg.wandb_experiment_name,
        run_name=experiment_name,
        config=wandb_config,
        use_logger=cfg.use_logger
    )
    
    accs = []
    for run in range(cfg.runs):
        log.info("Run {}/{}".format(run + 1, cfg.runs))
        
        # Initialize wandb run for this specific run
        if cfg.use_logger:
            wandb_logger.init_run(run, wandb_config)
        
        if cfg.model_name == "SAGE":
            model = SAGE(in_size, cfg.hidden_dim, out_size, cfg.num_layers).to(device)        
        # # model training
        log.info("Training...")
        acc = train(cfg, device, g, splits, model, num_classes, run, wandb_logger)
        accs.append(acc)
        
        # Log final results for this run
        wandb_logger.log_run_result(run, acc)
        
        # Finish this run
        if cfg.use_logger:
            wandb_logger.finish_run()

    # Log summary statistics
    wandb_logger.log_summary_statistics()

    mean_acc = torch.mean(torch.tensor(accs)).item()
    std_acc = torch.std(torch.tensor(accs)).item()
    
    print(f"Final Results:")
    print(f"Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

    return mean_acc

if __name__=='__main__':
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time / 60:.2f} minutes")
    print(f"Total execution time: {elapsed_time / 3600:.2f} hours")