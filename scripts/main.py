import inspect
import os

import torch

import cogdl.runner.options as options
from cogdl.runner.runner import Trainer
from cogdl.models import build_model
from cogdl.datasets import build_dataset
from cogdl.wrappers import fetch_model_wrapper, fetch_data_wrapper
from cogdl.utils import set_random_seed, tabulate_results
from tabulate import tabulate

def examine_link_prediction(args, dataset):
    if "link_prediction" in args.mw:
        args.num_entities = dataset.data.num_nodes
        # args.num_entities = len(torch.unique(self.data.edge_index))
        if dataset.data.edge_attr is not None:
            args.num_rels = len(torch.unique(dataset.data.edge_attr))
            args.monitor = "mrr"
        else:
            args.monitor = "auc"
    return args


def add_values_to_gnn_rec_args(args, dataset):
    args.data = dataset[0]
    args.data.apply(lambda x: x.to(args.devices))
    args.n_users = args.data.n_params["n_users"]
    args.n_items = args.data.n_params["n_items"]
    args.train_user_set = args.data.user_dict['train_user_set']
    args.adj_mat = args.data.norm_mat
    if args.gnn == "dgcf":
        # args.all_h_list = list(args.adj_mat.row)
        # args.all_t_list = list(args.adj_mat.col)
        args.n_train = args.data.n_params["n_train"]
    if args.gnn == 'dregn-cf':
        args.u_freqs = args.data.user_dict["u_freqs"]
        args.u_freqs = args.data.user_dict["u_freqs"]
    return args


def raw_experiment(args, model_wrapper_args, data_wrapper_args):
    # setup dataset and specify `num_features` and `num_classes` for model
    args.monitor = "val_acc"
    dataset = build_dataset(args)
    args.num_features = dataset.num_features
    if hasattr(args, "unsup") and args.unsup:
        args.num_classes = args.hidden_size
    else:
        args.num_classes = dataset.num_classes

    mw_class = fetch_model_wrapper(args.mw)
    dw_class = fetch_data_wrapper(args.dw)

    if mw_class is None:
        raise NotImplementedError("`model wrapper(--mw)` must be specified.")

    if dw_class is None:
        raise NotImplementedError("`data wrapper(--dw)` must be specified.")

    # unworthy code: share `args` between model and dataset_wrapper
    for key in inspect.signature(dw_class).parameters.keys():
        if hasattr(args, key) and key != "dataset":
            setattr(data_wrapper_args, key, getattr(args, key))
    # unworthy code: share `args` between model and model_wrapper
    for key in inspect.signature(mw_class).parameters.keys():
        if hasattr(args, key) and key != "model":
            setattr(model_wrapper_args, key, getattr(args, key))

    args = examine_link_prediction(args, dataset)

    if args.mw == "gnn_recommendation_mw":
        args = add_values_to_gnn_rec_args(args, dataset)

    # setup model
    model = build_model(args)
    # specify configs for optimizer
    optimizer_cfg = dict(
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_warmup_steps=args.n_warmup_steps,
        max_epoch=args.max_epoch,
        batch_size=args.batch_size if hasattr(args, "batch_size") else 0,
    )

    if hasattr(args, "hidden_size"):
        optimizer_cfg["hidden_size"] = args.hidden_size

    # setup model_wrapper
    if "embedding" in args.mw:
        model_wrapper = mw_class(model, **model_wrapper_args.__dict__)
    else:
        model_wrapper = mw_class(model, optimizer_cfg, **model_wrapper_args.__dict__)
    # setup data_wrapper
    dataset_wrapper = dw_class(dataset, **data_wrapper_args.__dict__)

    save_embedding_path = args.emb_path if hasattr(args, "emb_path") else None
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)

    # setup controller
    trainer = Trainer(
        max_epoch=args.max_epoch,
        device_ids=args.devices,
        cpu=args.cpu,
        save_embedding_path=save_embedding_path,
        cpu_inference=args.cpu_inference,
        # monitor=args.monitor,
        progress_bar=args.progress_bar,
        distributed_training=args.distributed,
        checkpoint_path=args.checkpoint_path,
    )

    # Go!!!
    result = trainer.run(model_wrapper, dataset_wrapper)
    return result


def run(args, model_wrapper_args, data_wrapper_args):
    print(
        f""" 
    |---------------------------------------------------{'-' * (len(args.mw) + len(args.dw))}|
     *** Using `{args.mw}` ModelWrapper and `{args.dw}` DataWrapper 
    |---------------------------------------------------{'-' * (len(args.mw) + len(args.dw))}|"""
    )
    results = []
    for seed in args.seed:
        set_random_seed(seed)
        out = raw_experiment(args, model_wrapper_args, data_wrapper_args)
        results.append(out)
    return results


def main():
    parser = options.get_training_parser()
    args, _ = parser.parse_known_args()
    args, model_wrapper_args, data_wrapper_args = options.parse_args_and_arch(parser, args)
    args.dataset = args.dataset[0]
    print(args)
    results = run(args, model_wrapper_args, data_wrapper_args)
    column_names = ["Variant"] + list(results[-1].keys())
    result_dict = {(args.model, args.dataset): results}
    result = tabulate_results(result_dict)
    print(tabulate(result, headers=column_names, tablefmt="github"))


if __name__ == "__main__":
    main()
