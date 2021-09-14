import copy
import datetime
from typing import Optional
import numpy as np
from tqdm import tqdm
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp

from cogdl.wrappers.data_wrapper.base_data_wrapper import DataWrapper
from cogdl.wrappers.model_wrapper.base_model_wrapper import ModelWrapper, EmbeddingModelWrapper
from cogdl.runner.runner_utils import evaluation_comp, load_model, save_model, ddp_end, ddp_after_epoch, Printer
from cogdl.runner.embed_runner import EmbeddingTrainer
from cogdl.runner.controller import DataController
from cogdl.data import Graph


def move_to_device(batch, device):
    if isinstance(batch, list) or isinstance(batch, tuple):
        if isinstance(batch, tuple):
            batch = list(batch)
        for i, x in enumerate(batch):
            if torch.is_tensor(x):
                batch[i] = x.to(device)
            elif isinstance(x, Graph):
                x.to(device)
    elif torch.is_tensor(batch) or isinstance(batch, Graph):
        batch = batch.to(device)
    return batch


def clip_grad_norm(params, max_norm):
    """Clips gradient norm."""
    if max_norm > 0:
        return torch.nn.utils.clip_grad_norm_(params, max_norm)
    else:
        return torch.sqrt(sum(p.grad.data.norm() ** 2 for p in params if p.grad is not None))


class Trainer(object):
    def __init__(
        self,
        max_epoch: int,
        nstage: int = 1,
        cpu: bool = False,
        checkpoint_path: str = "./checkpoints/checkpoint.pt",
        device_ids: Optional[list] = None,
        distributed_training: bool = False,
        distributed_inference: bool = False,
        master_addr: str = "localhost",
        master_port: int = 10086,
        monitor: str = "val_acc",
        early_stopping: bool = True,
        patience: int = 100,
        eval_step: int = 1,
        save_embedding_path: Optional[str] = None,
        cpu_inference: bool = False,
        progress_bar: str = "epoch",
        clip_grad_norm: float = 5.0,
    ):
        self.max_epoch = max_epoch
        self.nstage = nstage
        self.patience = patience
        self.early_stopping = early_stopping
        self.eval_step = eval_step
        self.monitor = monitor
        self.progress_bar = progress_bar

        self.cpu = cpu
        self.devices, self.world_size = self.set_device(device_ids)
        self.checkpoint_path = checkpoint_path

        self.distributed_training = distributed_training
        self.distributed_inference = distributed_inference
        self.master_addr = master_addr
        self.master_port = master_port

        self.cpu_inference = cpu_inference

        self.on_train_batch_transform = None
        self.on_eval_batch_transform = None
        self.clip_grad_norm = clip_grad_norm

        self.save_embedding_path = save_embedding_path

        self.data_controller = DataController(world_size=self.world_size, distributed=self.distributed_training)

        self.after_epoch_hooks = []
        self.pre_epoch_hooks = []
        self.training_end_hooks = []

        if distributed_training:
            self.register_training_end_hook(ddp_end)
            self.register_out_epoch_hook(ddp_after_epoch)

    def register_in_epoch_hook(self, hook):
        self.pre_epoch_hooks.append(hook)

    def register_out_epoch_hook(self, hook):
        self.after_epoch_hooks.append(hook)

    def register_training_end_hook(self, hook):
        self.training_end_hooks.append(hook)

    def set_device(self, device_ids: Optional[list]):
        """
        Return: devices, world_size
        """
        if device_ids is None or self.cpu:
            return [torch.device("cpu")], 0

        if isinstance(device_ids, int) and device_ids > 0:
            device_ids = [device_ids]
        elif isinstance(device_ids, list):
            pass
        else:
            raise ValueError("`device_id` has to be list of integers")
        if len(device_ids) == 0:
            return torch.device("cpu"), 0
        else:
            return [i for i in device_ids], len(device_ids)

    def run(self, model_w: ModelWrapper, dataset_w: DataWrapper):
        # for network/graph embedding models
        if isinstance(model_w, EmbeddingModelWrapper):
            return EmbeddingTrainer(self.save_embedding_path).run(model_w, dataset_w)

        # for deep learning models
        # set default loss_fn and evaluator for model_wrapper
        # mainly for in-cogdl setting
        model_w.default_loss_fn = dataset_w.get_default_loss_fn()
        model_w.default_evaluator = dataset_w.get_default_evaluator()
        if self.distributed_training and self.world_size > 1:
            self.dist_train(model_w, dataset_w)
        else:
            self.train(model_w, dataset_w, self.devices[0])

        best_model_w = load_model(model_w, self.checkpoint_path).to(self.devices[0])
        # disable `distributed` to inference once only
        final = self.test(best_model_w, dataset_w, self.devices[0])
        return final

    def dist_train(self, model_w: ModelWrapper, dataset_w: DataWrapper):
        mp.set_start_method("spawn", force=True)

        device_count = torch.cuda.device_count()
        if device_count < self.world_size:
            size = device_count
            print(f"Available device count ({device_count}) is less than world size ({self.world_size})")
        else:
            size = self.world_size

        print(f"Let's using {size} GPUs.")

        processes = []
        for rank in range(size):
            p = mp.Process(target=self.train, args=(model_w, dataset_w, rank))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    def build_optimizer(self, model_w):
        opt_wrap = model_w.setup_optimizer()
        if isinstance(opt_wrap, list) or isinstance(opt_wrap, tuple):
            assert len(opt_wrap) == 2
            optimizers, lr_schedulars = opt_wrap
        else:
            optimizers = opt_wrap
            lr_schedulars = None

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        if lr_schedulars and not isinstance(lr_schedulars, list):
            lr_schedulars = [lr_schedulars]
        return optimizers, lr_schedulars

    def initialize(self, model_w, rank=0, master_addr: str = "localhost", master_port: int = 10008):
        if self.distributed_training:
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = str(master_port)
            dist.init_process_group("nccl", rank=rank, world_size=self.world_size)
            model_w = copy.deepcopy(model_w).to(rank)
            model_w = DistributedDataParallel(model_w, device_ids=[rank])

            module = model_w.module
            model_w, model_ddp = module, model_w
            return model_w, model_ddp
        else:
            return model_w.to(rank), None

    def train(self, model_w, dataset_w, rank):
        model_w, model_aux = self.initialize(
            model_w, rank=rank, master_addr=self.master_addr, master_port=self.master_port
        )
        self.data_controller.prepare_data_wrapper(dataset_w, rank)

        optimizers, lr_schedulars = self.build_optimizer(model_w)
        best_index, compare_fn = evaluation_comp(self.monitor)
        best_model_w = None

        patience = 0
        for stage in range(self.nstage):
            with torch.no_grad():
                pre_stage_out = model_w.pre_stage(stage, dataset_w)
                dataset_w.pre_stage(stage, pre_stage_out)
                self.data_controller.training_proc_per_stage(dataset_w, rank)

            if self.progress_bar == "epoch":
                epoch_iter = tqdm(range(self.max_epoch))
                epoch_printer = Printer(epoch_iter.set_description, rank=rank, world_size=self.world_size)
            else:
                epoch_iter = range(self.max_epoch)
                epoch_printer = Printer(print, rank=rank, world_size=self.world_size)
            for epoch in epoch_iter:
                s_t = datetime.datetime.now()
                print_str_dict = dict()
                for hook in self.pre_epoch_hooks:
                    hook(self)

                # inductive setting ..
                dataset_w.train()
                train_loader = dataset_w.on_train_wrapper()
                training_loss = self.training_step(model_w, train_loader, optimizers, lr_schedulars, rank)
                print_str_dict["Epoch"] = epoch
                print_str_dict["TrainLoss"] = training_loss
                val_loader = dataset_w.on_val_wrapper()
                if val_loader is not None and (epoch % self.eval_step) == 0:
                    # inductive setting ..
                    dataset_w.eval()
                    # do validation in inference device
                    val_result = self.validate(model_w, dataset_w, rank)
                    print("val_result: ", val_result)
                    if val_result is not None:
                        monitoring = val_result[self.monitor]
                        if compare_fn(monitoring, best_index):
                            best_index = monitoring
                            patience = 0
                            best_model_w = model_w
                        else:
                            patience += 1
                            if self.early_stopping and patience >= self.patience:
                                break
                        # print_str_dict["ValMetric"] = monitoring
                        print_str_dict["recall"] = val_result["recall"]
                        print_str_dict["ndcg"] = val_result["ndcg"]
                        print_str_dict["precision"] = val_result["precision"]
                        print_str_dict["hit_ratio"] = val_result["hit_ratio"]
                        # print_str_dict["val_acc"] = val_result["val_acc"]
                epoch_printer(print_str_dict)

                for hook in self.after_epoch_hooks:
                    hook(self)

                e_t = datetime.datetime.now()
                print(" time of a epoch: ", e_t - s_t)
            with torch.no_grad():
                post_stage_out = model_w.post_stage(stage, dataset_w)
                dataset_w.post_stage(stage, post_stage_out)

            if best_model_w is None:
                best_model_w = model_w

        save_model(best_model_w.to("cpu"), self.checkpoint_path)
        for hook in self.training_end_hooks:
            hook(self)

    def validate(self, model_w: ModelWrapper, dataset_w: DataWrapper, device):
        # ------- distributed training ---------
        if self.distributed_training:
            return self.distributed_test(model_w, dataset_w, device)
        # ------- distributed training ---------

        model_w.eval()
        if self.cpu_inference:
            model_w.to("cpu")
            _device = device
        else:
            _device = device

        val_loader = dataset_w.on_val_wrapper()
        result = self.val_step(model_w, val_loader, _device)

        model_w.to(device)
        return result

    def test(self, model_w: ModelWrapper, dataset_w: DataWrapper, device):
        model_w.eval()
        if self.cpu_inference:
            model_w.to("cpu")
            _device = device
        else:
            _device = device

        test_loader = dataset_w.on_test_wrapper()
        result = self.test_step(model_w, test_loader, _device)

        model_w.to(device)
        return result

    def distributed_test(self, model_w: ModelWrapper, dataset_w: DataWrapper, rank):
        model_w.eval()
        if rank == 0:
            if self.cpu_inference:
                model_w.to("cpu")
                _device = rank
            else:
                _device = rank
            test_loader = dataset_w.on_test_wrapper()
            result = self.test_step(model_w, test_loader, _device)
            object_list = [result]
            model_w.to(rank)
        else:
            object_list = [None, None]
        dist.broadcast_object_list(object_list, src=0)
        result = object_list[0]
        return result

    def training_step(self, model_w, train_loader, optimizers, lr_schedulars, device):
        model_w.train()
        losses = []

        if self.progress_bar == "iteration":
            train_loader = tqdm(train_loader)
        count=0
        for batch in train_loader:
            count += 1
            # batch = batch.to(device)
            batch = move_to_device(batch, device)
            loss = model_w.on_train_step(batch)

            for optimizer in optimizers:
                optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model_w.parameters(), self.clip_grad_norm)

            for optimizer in optimizers:
                optimizer.step()

            losses.append(loss.item())
        print("count: ", count)
        if lr_schedulars is not None:
            for lr_schedular in lr_schedulars:
                lr_schedular.step()
        return np.mean(losses)

    @torch.no_grad()
    def val_step(self, model_w, val_loader, device):
        model_w.eval()
        result = {'precision': 0.,
                  'recall': 0.,
                  'ndcg': 0.,
                  'hit_ratio': 0.,
                  'val_acc': 0.}
        for index, batch in enumerate(val_loader):
            # batch = batch.to(device)
            batch = move_to_device(batch, device)
            model_w.on_val_step(batch)
            out = model_w.collect_notes()
            # print("index --- valid out: ", index, out)
            result['precision'] += out['precision']
            result['recall'] += out['recall']
            result['ndcg'] += out['ndcg']
            result['hit_ratio'] += out['hit_ratio']
            result['val_acc'] += out['val_acc']
            # print("index --- valid result: ", index, result)
        model_w.note("precision", result["precision"])
        model_w.note("recall", result["recall"])
        model_w.note("ndcg", result["ndcg"])
        model_w.note("hit_ratio", result["hit_ratio"])
        model_w.note("val_acc", result["val_acc"])
        return model_w.collect_notes()

    @torch.no_grad()
    def test_step(self, model_w, test_loader, device):
        model_w.eval()
        result = {'precision': 0.,
                  'recall': 0.,
                  'ndcg': 0.,
                  'hit_ratio': 0.,
                  'val_acc': 0.}
        for index, batch in enumerate(test_loader):
            # batch = batch.to(device)
            batch = move_to_device(batch, device)
            model_w.on_test_step(batch)
            out = model_w.collect_notes()
            # print("test out: ", out)
            result['precision'] += out['precision']
            result['recall'] += out['recall']
            result['ndcg'] += out['ndcg']
            result['hit_ratio'] += out['hit_ratio']
            result['val_acc'] += out['val_acc']
            # print("index --- test result: ", index, result)
        model_w.note("precision", result["precision"])
        model_w.note("recall", result["recall"])
        model_w.note("ndcg", result["ndcg"])
        model_w.note("hit_ratio", result["hit_ratio"])
        model_w.note("val_acc", result["val_acc"])
        return model_w.collect_notes()

    def distributed_model_proc(self, model_w: ModelWrapper, rank):
        _model = model_w.wrapped_model
        ddp_model = DistributedDataParallel(_model, device_ids=[rank])
        # _model = ddp_model.module
        model_w.wrapped_model = ddp_model
