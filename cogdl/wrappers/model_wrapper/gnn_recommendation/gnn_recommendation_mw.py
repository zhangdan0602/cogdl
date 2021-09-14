# ! /usr/bin/python
# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         gnn_recommendation_mw
# Description:
# Author:       Zd
# Date:         2021/9/7
# -------------------------------------------------------------------------------

from .. import ModelWrapper, register_model_wrapper
import numpy as np
import scipy.sparse as sp
import torch
import multiprocessing as mp
import random
import os
import datetime
import heapq
from sklearn.metrics import roc_auc_score
from cogdl.datasets import build_dataset
from cogdl.models import build_model

@register_model_wrapper("gnn_recommendation_mw")
class GNNRecommendationModelWrapper(ModelWrapper):
	@staticmethod
	def add_args(parser):
		# ===== dataset ===== #
		parser.add_argument("--dataset", nargs="?", default="amazon",
							help="Choose a dataset:[amazon,yelp2018,ali,aminer]")
		# ===== train ===== # 
		parser.add_argument("--gnn", nargs="?", default="lightgcn",
							help="Choose a recommender:[lightgcn, ngcf, dgcf, dregn-cf, gcmc]")
		parser.add_argument('--test_flag', nargs='?', default='part',
							help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

	def __init__(self, model, optimizer_cfg, gnn, test_flag):
		super(GNNRecommendationModelWrapper, self).__init__()
		self.model = model
		self.optimizer_config = optimizer_cfg
		self.gnn = gnn
		self.batch_test_flag = True
		self.batch_size = 1024
		self.Ks = [20, 50]
		self.test_flag = test_flag

	def train_step(self, batch):
		train_s_t = datetime.datetime.now()
		batch_loss, _, _ = self.model(batch)
		# print('batch_loss:', batch_loss.item())
		train_e_t = datetime.datetime.now()
		# print("time of training: ", train_e_t - train_s_t)
		return batch_loss

	def test_step(self, batch):
		result = {'precision': np.zeros(len(self.Ks)),
				  'recall': np.zeros(len(self.Ks)),
				  'ndcg': np.zeros(len(self.Ks)),
				  'hit_ratio': np.zeros(len(self.Ks)),
				  'val_acc': 0.}
		test_s_t = datetime.datetime.now()
		batch_results = []
		count = 0
		user_batch = batch[0].cpu()[0].numpy().tolist()
		n_items = batch[1].item()
		n_test_users = batch[2].item()
		# print("n_test_users: ", n_test_users)
		train_user_set = batch[3]
		for k, v in train_user_set.items():
			r = []
			for val in v:
				r.append(val.item())
			train_user_set[k] = r
		test_user_set = batch[4]
		for k, v in test_user_set.items():
			r = []
			for val in v:
				r.append(val.item())
			test_user_set[k] = r
		user_list_batch = [i.item() for i in batch[5]]
		user_gcn_emb, item_gcn_emb = self.model.generate()
		u_g_embeddings = user_gcn_emb[user_batch]
		i_batch_size = self.batch_size
		if self.batch_test_flag:
			# batch-item test
			n_item_batchs = n_items // i_batch_size + 1
			rate_batch = np.zeros(shape=(len(user_batch), n_items))
			i_count = 0
			for i_batch_id in range(n_item_batchs):
				i_start = i_batch_id * i_batch_size
				i_end = min((i_batch_id + 1) * i_batch_size, n_items)

				item_batch = torch.LongTensor(np.array(range(i_start, i_end))).view(i_end - i_start).to(self.device)
				i_g_embddings = item_gcn_emb[item_batch]

				i_rate_batch = self.model.rating(u_g_embeddings, i_g_embddings).detach().cpu()
				# i_rate_batch = self.model.rating(u_g_embeddings, i_g_embddings).detach()
				# i_rate_batch = torch.squeeze(i_rate_batch).cpu()
				rate_batch[:, i_start:i_end] = i_rate_batch
				i_count += i_rate_batch.shape[1]

			assert i_count == n_items
		else:
			# all-item test
			item_batch = torch.LongTensor(np.array(range(0, n_items))).view(n_items, -1).to(self.device)
			i_g_embddings = item_gcn_emb[item_batch]
			rate_batch = self.model.rating(u_g_embeddings, i_g_embddings).detach().cpu()

		user_batch_rating_uid = []  # zip(rate_batch, user_list_batch, [self.Ks] * len(rate_batch))
		for rate, user in zip(rate_batch, user_list_batch):
			user_batch_rating_uid.append(
				[
					rate,
					train_user_set[user] if user in train_user_set else [],
					test_user_set[user],
					self.Ks,
					n_items,
				]
			)
		for x in user_batch_rating_uid:
			batch_results.append(self.test_one_user(x))
		count += len(batch_results)
		for re in batch_results:
			result['precision'] += re['precision'] / n_test_users
			result['recall'] += re['recall'] / n_test_users
			result['ndcg'] += re['ndcg'] / n_test_users
			result['hit_ratio'] += re['hit_ratio'] / n_test_users
			result['val_acc'] += re['val_acc'] / n_test_users
		test_e_t = datetime.datetime.now()
		# print("time of test: ", test_e_t - test_s_t)
		# print("test result: ", result)
		self.note("precision", result["precision"][0])
		self.note("recall", result["recall"][0])
		self.note("ndcg", result["ndcg"][0])
		self.note("hit_ratio", result["hit_ratio"][0])
		self.note("val_acc", result["val_acc"])

	def val_step(self, batch):
		valid_s_t = datetime.datetime.now()
		result = self.test_step(batch)
		valid_e_t = datetime.datetime.now()
		if result is not None:
			print("time of valid: ", valid_e_t - valid_s_t)
			self.note("precision", result["precision"][0])
			self.note("recall", result["recall"][0])
			self.note("ndcg", result["ndcg"][0])
			self.note("hit_ratio", result["hit_ratio"][0])
			self.note("val_acc", result["val_acc"])

	def setup_optimizer(self):
		cfg = self.optimizer_config
		if hasattr(self.model, "setup_optimizer"):
			model_spec_optim = self.model.setup_optimizer(cfg)
			if model_spec_optim is not None:
				return model_spec_optim
		return torch.optim.Adam(self.model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

	def test_one_user(self, x):
		rating = x[0]
		training_items = x[1]
		user_pos_test = x[2]
		self.Ks = x[3]
		self.n_items = x[4]
		all_items = set(range(0, self.n_items))
		test_items = list(all_items - set(training_items))

		if self.test_flag == 'part':
			r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, self.Ks)
		else:
			r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, self.Ks)

		return get_performance(user_pos_test, r, auc, self.Ks)


def recall(rank, ground_truth, N):
	return len(set(rank[:N]) & set(ground_truth)) / float(len(set(ground_truth)))


def precision_at_k(r, k):
	"""Score is precision @ k
	Relevance is binary (nonzero is relevant).
	Returns:
		Precision @ k
	Raises:
		ValueError: len(r) must be >= k
	"""
	assert k >= 1
	r = np.asarray(r)[:k]
	return np.mean(r)


def average_precision(r, cut):
	"""Score is average precision (area under PR curve)
	Relevance is binary (nonzero is relevant).
	Returns:
		Average precision
	"""
	r = np.asarray(r)
	out = [precision_at_k(r, k + 1) for k in range(cut) if r[k]]
	if not out:
		return 0.0
	return np.sum(out) / float(min(cut, np.sum(r)))


def mean_average_precision(rs):
	"""Score is mean average precision
	Relevance is binary (nonzero is relevant).
	Returns:
		Mean average precision
	"""
	return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=1):
	"""Score is discounted cumulative gain (dcg)
	Relevance is positive real values.  Can use binary
	as the previous methods.
	Returns:
		Discounted cumulative gain
	"""
	r = np.asfarray(r)[:k]
	if r.size:
		if method == 0:
			return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
		elif method == 1:
			return np.sum(r / np.log2(np.arange(2, r.size + 2)))
		else:
			raise ValueError("method must be 0 or 1.")
	return 0.0


def ndcg_at_k(r, k, ground_truth, method=1):
	"""Score is normalized discounted cumulative gain (ndcg)
	Relevance is positive real values.  Can use binary
	as the previous methods.
	Returns:
		Normalized discounted cumulative gain

		Low but correct defination
	"""
	GT = set(ground_truth)
	if len(GT) > k:
		sent_list = [1.0] * k
	else:
		sent_list = [1.0] * len(GT) + [0.0] * (k - len(GT))
	dcg_max = dcg_at_k(sent_list, k, method)
	if not dcg_max:
		return 0.0
	return dcg_at_k(r, k, method) / dcg_max


def recall_at_k(r, k, all_pos_num):
	# if all_pos_num == 0:
	#     return 0
	r = np.asfarray(r)[:k]
	return np.sum(r) / all_pos_num


def hit_at_k(r, k):
	r = np.array(r)[:k]
	if np.sum(r) > 0:
		return 1.0
	else:
		return 0.0


def F1(pre, rec):
	if pre + rec > 0:
		return (2.0 * pre * rec) / (pre + rec)
	else:
		return 0.0


def AUC(ground_truth, prediction):
	try:
		res = roc_auc_score(y_true=ground_truth, y_score=prediction)
	except Exception:
		res = 0.0
	return res


def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
	item_score = {}
	for i in test_items:
		item_score[i] = rating[i]

	K_max = max(Ks)
	K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

	r = []
	for i in K_max_item_score:
		if i in user_pos_test:
			r.append(1)
		else:
			r.append(0)
	auc = 0.0
	return r, auc


def get_auc(item_score, user_pos_test):
	item_score = sorted(item_score.items(), key=lambda kv: kv[1])
	item_score.reverse()
	item_sort = [x[0] for x in item_score]
	posterior = [x[1] for x in item_score]

	r = []
	for i in item_sort:
		if i in user_pos_test:
			r.append(1)
		else:
			r.append(0)
	auc = AUC(ground_truth=r, prediction=posterior)
	return auc


def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
	item_score = {}
	for i in test_items:
		item_score[i] = rating[i]

	K_max = max(Ks)
	K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

	r = []
	for i in K_max_item_score:
		if i in user_pos_test:
			r.append(1)
		else:
			r.append(0)
	auc = get_auc(item_score, user_pos_test)
	return r, auc


def get_performance(user_pos_test, r, auc, Ks):
	precision, recall, ndcg, hit_ratio = [], [], [], []

	for K in Ks:
		precision.append(precision_at_k(r, K))
		recall.append(recall_at_k(r, K, len(user_pos_test)))
		ndcg.append(ndcg_at_k(r, K, user_pos_test))
		hit_ratio.append(hit_at_k(r, K))

	return {
		"recall": np.array(recall),
		"precision": np.array(precision),
		"ndcg": np.array(ndcg),
		"hit_ratio": np.array(hit_ratio),
		"val_acc": auc,
	}


def early_stopping(log_value, best_value, stopping_step, expected_order="acc", flag_step=100):
	# early stopping strategy:
	assert expected_order in ["acc", "dec"]

	if (expected_order == "acc" and log_value >= best_value) or (expected_order == "dec" and log_value <= best_value):
		stopping_step = 0
		best_value = log_value
	else:
		stopping_step += 1

	if stopping_step >= flag_step:
		print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
		should_stop = True
	else:
		should_stop = False
	return best_value, stopping_step, should_stop
