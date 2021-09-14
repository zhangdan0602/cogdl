import datetime
import heapq
import multiprocessing as mp
import os
import random
from time import time

import numpy as np
import torch
from prettytable import PrettyTable

from cogdl.datasets import build_dataset
from cogdl.models import build_model
from sklearn.metrics import roc_auc_score

from . import BaseTask, register_task


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
		"auc": auc,
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
		print("Early stopping is trigger at step: {}, stop at: {} log:{}".format(flag_step, stopping_step, log_value))
		should_stop = True
	else:
		should_stop = False
	return best_value, stopping_step, should_stop


@register_task("recommendation_archi")
class Recommendation(BaseTask):
	@staticmethod
	def add_args(parser):
		# ===== dataset ===== #
		parser.add_argument("--dataset", nargs="?", default="amazon",
							help="Choose a dataset:[amazon,yelp2018,ali,aminer]")
		parser.add_argument(
			"--data_path", nargs="?", default="data/", help="Input data path."
		)

		# ===== train ===== # 
		parser.add_argument("--gnn", nargs="?", default="lightgcn",
							help="Choose a recommender:[lightgcn, ngcf, dgcf, dregn-cf, gcmc, agcn]")
		parser.add_argument('--epoch', type=int, default=1000, help='number of epochs')
		parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
		parser.add_argument('--test_batch_size', type=int, default=2048, help='batch size in evaluation phase')
		parser.add_argument('--dim', type=int, default=64, help='embedding size')
		parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization weight, 1e-5 for NGCF')
		parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
		parser.add_argument("--mess_dropout", type=bool, default=False, help="consider mess dropout or not")
		parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of mess dropout")
		parser.add_argument("--edge_dropout", type=bool, default=False, help="consider edge dropout or not")
		parser.add_argument("--edge_dropout_rate", type=float, default=0.1, help="ratio of edge sampling")
		parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")

		parser.add_argument("--ns", type=str, default='rns', help="rns,mixgcf")
		parser.add_argument("--K", type=int, default=1, help="number of negative in K-pair loss")

		parser.add_argument("--n_negs", type=int, default=1, help="number of candidate negative")
		parser.add_argument("--pool", type=str, default='concat', help="[concat, mean, sum, final]")

		parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
		parser.add_argument("--gpu_id", type=int, default=2, help="gpu id")
		parser.add_argument('--Ks', nargs='?', default='[20]', help='Output sizes of every layer')
		parser.add_argument('--test_flag', nargs='?', default='part',
							help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

		parser.add_argument("--context_hops", type=int, default=3, help="hop")

		# ===== test ===== #
		parser.add_argument("--train_method", type=str, default='offline', help="[init, offline, nearline]")
		parser.add_argument("--u_ids", default='[0]', help="test u_id")
		# ===== save model ===== #
		parser.add_argument("--save", type=bool, default=True, help="save model or not")
		parser.add_argument(
			"--out_dir", type=str, default="./weights", help="output directory for model"
		)

	def __init__(self, args, dataset=None, model=None):
		super(Recommendation, self).__init__(args)
		"""fix the random seed"""
		seed = 2020
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

		"""read args"""
		os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
		# self.device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
		self.device = args.gpu_id if args.cuda else torch.device("cpu")
		print('rec device: ', self.device)
		self.epoch = args.epoch
		self.batch_size = args.batch_size
		self.n_negs = args.n_negs
		self.test_flag = args.test_flag
		self.batch_test_flag = args.batch_test_flag
		self.save = args.save
		self.out_dir = args.out_dir
		self.Ks = eval(args.Ks)
		self.K = args.K
		self.u_ids = eval(args.u_ids)
		self.num_workers = mp.cpu_count() // 2
		self.gnn = args.gnn
		self.train_method = args.train_method
		"""build dataset"""
		dataset = build_dataset(args) if dataset is None else dataset
		self.data = dataset[0]
		self.data.apply(lambda x: x.to(self.device))
		args.n_users = self.data.n_params["n_users"]
		args.n_items = self.data.n_params["n_items"]
		args.train_user_set = self.data.user_dict['train_user_set']

		self.n_users = args.n_users
		self.n_items = args.n_items
		args.adj_mat = self.data.norm_mat
		self.dataset_name = args.dataset
		args.device = self.device
		self.u_id = eval(args.u_ids)
		if self.gnn == 'dgcf':
			# args.all_h_list = list(args.adj_mat.row)
			# args.all_t_list = list(args.adj_mat.col)
			args.n_train = self.data.n_params["n_train"]
		if self.gnn == 'dregn-cf':
			args.u_freqs = self.data.user_dict["u_freqs"]
			self.u_freqs = args.data.user_dict["u_freqs"]

		"""build model"""
		# self.device = [0, 1, 2, 3, 4, 5, 6, 7]
		model = build_model(args) if model is None else model
		# self.model = torch.nn.DataParallel(model.cuda(), device_ids=self.device, output_device=5)
		self.model = model.to(self.device)
		self.model.set_device(self.device)
		"""define optimizer"""
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

	def train(self, unittest=False):
		if self.train_method == "init":
			res = self.init_train(unittest)
			return res
		elif self.train_method == "offline":
			res = self.offline_train()
			return res
		elif self.train_method == "nearline":
			res = self.nearline_predict()
			return res

	def init_train(self, unittest=False):
		cur_best_pre_0, stopping_step, best_value, best_epoch = 0, 0, 0, 0
		print("start training ...", datetime.datetime.now())
		res = {}
		for epoch in range(self.epoch):
			# shuffle training data
			# print("\nepoch: ", epoch)
			train_cf = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in self.data.train_cf], np.int32))
			train_cf_ = train_cf
			index = np.arange(len(train_cf_))
			# np.random.shuffle(index)
			train_cf_ = train_cf_[index].to(self.device)
			"""training"""
			self.model.train()
			loss, s = 0, 0
			train_s_t = time()
			loss, s = self._train_step(loss, s, train_cf, train_cf_)
			train_e_t = time()
			# print('loss:', loss.item())
			if epoch % 5 == 0:
				"""testing"""
				train_res = PrettyTable()
				train_res.field_names = ["Epoch", "training time(s)", "tesing time(s)", "Loss", "recall", "ndcg",
										 "precision", "hit_ratio"]
				self.model.eval()
				test_s_t = time()
				test_ret = self._test_step(split="test", unittest=False)
				test_e_t = time()
				train_res.add_row(
					[epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), test_ret['recall'],
					 test_ret['ndcg'],
					 test_ret['precision'], test_ret['hit_ratio']])
				if self.data.user_dict['valid_user_set'] is None:
					valid_ret = test_ret
				else:
					test_s_t = time()
					valid_ret = self._test_step(split="valid", unittest=False)
					test_e_t = time()
					train_res.add_row(
						[epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), valid_ret['recall'],
						 valid_ret['ndcg'],
						 valid_ret['precision'], valid_ret['hit_ratio']])
				print(train_res)

				# *********************************************************
				# early stopping when cur_best_pre_0 is decreasing for 10 successive steps.
				cur_best_pre_0, stopping_step, should_stop = early_stopping(valid_ret['recall'][0],
																			cur_best_pre_0,
																			stopping_step,
																			expected_order='acc',
																			flag_step=10)

				"""save weight"""
				if valid_ret['recall'][0] == cur_best_pre_0 and self.save:
					best_epoch = epoch
					best_value = cur_best_pre_0
					res = {"recall": valid_ret["recall"][0], "ndcg": valid_ret["ndcg"][0],
						   "precision": valid_ret["precision"][0], "hit_ratio": valid_ret["hit_ratio"][0]}
					day = datetime.datetime.today()
					print(day, type(day))
					# dir = os.getcwd() + self.out_dir + self.dataset_name
					if not os.path.exists(self.out_dir):
						os.mkdir(self.out_dir)
					torch.save(self.model.state_dict(),
							   self.out_dir + ('/Epoch-%d-%f.pt' % (epoch, cur_best_pre_0)))
				if should_stop:
					break
			else:
				# logging.info('training loss at epoch %d: %f' % (epoch, loss.item()))
				print('using time %.4fs, training loss at epoch %d: %.4f' % (train_e_t - train_s_t, epoch, loss.item()))
		print('early stopping at %d, recall@20:%.4f, best epoch: %d' % (best_epoch, cur_best_pre_0, best_epoch))
		return res

	def offline_train(self):
		train_s = time()
		loss, s = 0, 0
		print(self.data.n_params['n_items'])
		for epoch in range(self.epoch):
			# shuffle training data
			train_s_t = time()
			train_cf = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in self.data.train_cf], np.int32))
			train_cf_ = train_cf
			index = np.arange(len(train_cf_))
			# np.random.shuffle(index)
			train_cf_ = train_cf_[index].to(self.device)
			"""training"""
			self.model.train()
			loss, s = self._train_step(loss, s, train_cf, train_cf_)
			train_e_t = time()
			print('train time: ', train_e_t - train_s_t, 'loss: ', loss.item())
		train_e = time()
		last_day = datetime.datetime.today().date()
		if not os.path.exists(self.out_dir):
			os.mkdir(self.out_dir)
		torch.save(self.model.state_dict(),
				   self.out_dir + '/' + str(last_day) + '.pt')
		train_res = {"Epoch": self.epoch, "training time(s)": train_e - train_s, "loss": loss.item()}
		return train_res

	def nearline_predict(self):
		result = {}
		today = datetime.date.today()
		load_pt = self.out_dir + '/' + str(today) + '.pt'
		if not os.path.exists(load_pt):
			last_day = datetime.date.today() - datetime.timedelta(days=1)
			load_pt = self.out_dir + '/' + str(last_day) + '.pt'
		print(load_pt)
		pt = torch.load(load_pt)
		self.model.load_state_dict(pt)
		self.model.eval()
		n_items = self.data.n_params['n_items']
		user_gcn_emb, item_gcn_emb = self.model.generate()
		user_batch = torch.LongTensor(np.array([self.u_ids])).to(self.device)
		u_g_embeddings = user_gcn_emb[user_batch]
		item_batch = torch.LongTensor(np.array(range(0, n_items))).view(n_items, -1).squeeze().to(self.device)
		i_g_embddings = item_gcn_emb[item_batch]
		rate_batch = self.model.rating(u_g_embeddings, i_g_embddings.squeeze()).detach().cpu()
		train_user_set = self.data.user_dict["train_user_set"]

		def ranklist_by_sorted(u_id, test_items, rating, Ks):
			item_score = {}
			for i in test_items:
				item_score[i] = rating[i]
			K_max = max(Ks)
			K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)
			# r = []
			one_res = {}
			for i in K_max_item_score:
				if i in test_items:
					one_res[i] = rating[i].item()
				# r.append(i)
			return one_res

		for rate, u_id in zip(rate_batch.squeeze(), self.u_ids):
			training_items = train_user_set[u_id] if u_id in train_user_set else []
			all_items = set(range(0, self.n_items))
			test_items = list(all_items - set(training_items))
			one_res = ranklist_by_sorted(u_id, test_items, rate, self.Ks)
			if u_id not in result:
				result[u_id] = one_res
		return result

	def _train_step(self, loss, s, train_cf, train_cf_):
		while s + self.batch_size <= len(train_cf):
			batch = self.get_feed_dict(train_cf_,
									   self.data.user_dict['train_user_set'],
									   s,
									   s + self.batch_size,
									   self.data.n_params["n_items"],
									   self.n_negs,
									   self.device)
			# print('get batch done.')
			batch_loss, _, _ = self.model(batch)

			self.optimizer.zero_grad()
			batch_loss.backward()
			self.optimizer.step()

			loss += batch_loss
			s += self.batch_size
		print('loss:', loss.item())
		return loss, s

	def _test_step(self, split="val", unittest=False):
		result = {'precision': np.zeros(len(self.Ks)),
				  'recall': np.zeros(len(self.Ks)),
				  'ndcg': np.zeros(len(self.Ks)),
				  'hit_ratio': np.zeros(len(self.Ks)),
				  'auc': 0.}

		n_items = self.data.n_params['n_items']
		train_user_set = self.data.user_dict["train_user_set"]
		if split == 'test':
			test_user_set = self.data.user_dict['test_user_set']
		else:
			test_user_set = self.data.user_dict['valid_user_set']
			if test_user_set is None:
				test_user_set = self.data.user_dict['test_user_set']
		# print(len(test_user_set))
		pool = mp.Pool(self.num_workers)

		u_batch_size = self.batch_size
		i_batch_size = self.batch_size

		test_users = list(test_user_set.keys())
		n_test_users = len(test_users)
		n_user_batchs = n_test_users // u_batch_size + 1

		count = 0

		user_gcn_emb, item_gcn_emb = self.model.generate()

		for u_batch_id in range(n_user_batchs):
			start = u_batch_id * u_batch_size
			end = (u_batch_id + 1) * u_batch_size

			user_list_batch = test_users[start:end]
			user_batch = torch.LongTensor(np.array(user_list_batch)).to(self.device)
			u_g_embeddings = user_gcn_emb[user_batch]

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
			batch_results = []
			for x in user_batch_rating_uid:
				batch_results.append(self.test_one_user(x))
			# batch_result = pool.map(self.test_one_user, user_batch_rating_uid)
			count += len(batch_results)

			for re in batch_results:
				result['precision'] += re['precision'] / n_test_users
				result['recall'] += re['recall'] / n_test_users
				result['ndcg'] += re['ndcg'] / n_test_users
				result['hit_ratio'] += re['hit_ratio'] / n_test_users
				result['auc'] += re['auc'] / n_test_users

		assert count == n_test_users
		pool.close()
		# print(result)
		return result

	def test_one_user(self, x):
		# # user u's ratings for user u
		# rating = x[0]
		# # uid
		# u = x[1]
		# print(rating, u)
		# # user u's items in the training set
		# try:
		# 	training_items = self.data.user_dict["train_user_set"][u]
		# except Exception:
		# 	training_items = []
		# # user u's items in the test set
		# user_pos_test = self.data.user_dict["test_user_set"][u]
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

	def get_feed_dict(self, train_entity_pairs, train_pos_set, start, end, n_items, n_negs=1, device="cpu"):
		device = self.device

		def sampling(user_item, train_set, n):
			neg_items = []
			for user, _ in user_item.cpu().numpy():
				user = int(user)
				negitems = []
				for i in range(n):  # sample n times
					while True:
						negitem = random.choice(range(n_items))
						if negitem not in train_set[user]:
							break
					negitems.append(negitem)
				neg_items.append(negitems)
			return neg_items

		feed_dict = {}
		entity_pairs = train_entity_pairs[start:end]
		feed_dict["users"] = entity_pairs[:, 0]
		feed_dict["pos_items"] = entity_pairs[:, 1]
		feed_dict["neg_items"] = torch.LongTensor(sampling(entity_pairs, train_pos_set, n_negs * self.K)).to(device)
		# print(feed_dict["users"])
		# print(feed_dict["pos_items"])
		# print(feed_dict["neg_items"])
		users = entity_pairs[:, 0].cpu().numpy()
		pos_items = np.concatenate([train_pos_set[u] for u in users])
		# print(len(users), len(pos_items))
		items = np.unique(pos_items)
		item_position_dict = dict(zip(items, range(len(items))))
		get_item_position = lambda i: item_position_dict.get(int(i), np.nan)
		item_pos_inds = np.vectorize(get_item_position)(pos_items)
		user_pos_inds = np.concatenate(
			[[idx] * len(train_pos_set[u]) for idx, (u) in enumerate(users)]
		)
		user_pos_inds = user_pos_inds[~np.isnan(item_pos_inds)]
		item_pos_inds = item_pos_inds[~np.isnan(item_pos_inds)]
		users = torch.from_numpy(users)
		items = torch.from_numpy(items)
		user_pos_inds = torch.from_numpy(user_pos_inds)
		item_pos_inds = torch.from_numpy(item_pos_inds)
		feed_dict["users1"] = users
		feed_dict["items"] = items
		feed_dict["user_pos_inds"] = user_pos_inds
		feed_dict["item_pos_inds"] = item_pos_inds

		return feed_dict
