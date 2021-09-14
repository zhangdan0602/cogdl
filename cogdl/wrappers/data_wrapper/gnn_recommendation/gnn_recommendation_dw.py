# ! /usr/bin/python
# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         gnn_recommendation_dw
# Description:
# Author:       Zd
# Date:         2021/9/7
# -------------------------------------------------------------------------------


from .. import DataWrapper, register_data_wrapper
from collections import defaultdict, Counter
import torch
import numpy as np
import random
import multiprocessing as mp
from torch.utils.data import Dataset, DataLoader

from ... import ModelWrapper


@register_data_wrapper("gnn_recommendation_dw")
class GNNRecommendationDataWrapper(DataWrapper):
	def __init__(self, dataset):
		super(GNNRecommendationDataWrapper, self).__init__(dataset)
		self.dataset = dataset
		self.device = "cuda:0"
		self.batch_size = 1024
		self.n_negs = 1

	def train_wrapper(self):
		data = self.dataset[0]
		train_cf = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in data.train_cf], np.int32))
		train_cf_ = train_cf
		index = np.arange(len(train_cf_))
		train_cf_ = train_cf_[index].to(self.device)
		s = 0
		train_data= []
		while(s + self.batch_size <= len(train_cf)):
			batch = self.get_feed_dict(train_cf_,
								   data.user_dict['train_user_set'],
								   s,
								   s + self.batch_size,
								   data.n_params["n_items"],
								   self.n_negs,
								   self.device)
			train_data.append(batch)
			s += self.batch_size
		train_data = My_Dataset(train_data)
		data_loader = DataLoader(train_data, batch_size=1, shuffle=False)
		return data_loader

	def val_wrapper(self):
		self.data = self.dataset[0]
		if self.data.user_dict['valid_user_set'] is None:
			data_loader = None
		else:
			valid_user_set = self.data.user_dict['valid_user_set']
			data_loader = self.get_test_data_loader(valid_user_set)
		return data_loader

	def test_wrapper(self):
		test_user_set = self.data.user_dict['test_user_set']
		data_loader = self.get_test_data_loader(test_user_set)
		return data_loader

	def get_feed_dict(self, train_entity_pairs, train_pos_set, start, end, n_items, n_negs=1, device="cuda:0"):
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
		K = 1
		feed_dict = {}
		entity_pairs = train_entity_pairs[start:end]
		feed_dict["users"] = entity_pairs[:, 0]
		feed_dict["pos_items"] = entity_pairs[:, 1]
		feed_dict["neg_items"] = torch.LongTensor(sampling(entity_pairs, train_pos_set, n_negs * K)).to(device)
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

	def get_test_data_loader(self, test_user_set):
		test_data = []
		self.data = self.dataset[0]
		n_items = self.data.n_params['n_items']
		train_user_set = self.data.user_dict["train_user_set"]
		print("len(test_user_set): ", len(test_user_set))
		u_batch_size = self.batch_size
		test_users = list(test_user_set.keys())
		n_test_users = len(test_users)
		n_user_batchs = n_test_users // u_batch_size + 1

		for u_batch_id in range(n_user_batchs):
			start = u_batch_id * u_batch_size
			end = (u_batch_id + 1) * u_batch_size
			user_list_batch = test_users[start:end]
			user_batch = torch.LongTensor(np.array(user_list_batch)).to(self.device)
			test_data.append([user_batch, n_items, n_test_users, train_user_set, test_user_set, user_list_batch])

		test_data = My_Dataset(test_data)
		data_loader = DataLoader(test_data, batch_size=1, shuffle=False)
		return data_loader

class My_Dataset(Dataset):
	def __init__(self, batch_data):
		super(My_Dataset, self).__init__()
		self.data = batch_data

	def __getitem__(self, index):
		return self.data[index]

	def __len__(self):
		return len(self.data)


