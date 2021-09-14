# ! /usr/bin/python
# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         lr_gccf
# Description:
# Author:       Zd
# Date:         2021/7/1
# -------------------------------------------------------------------------------
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
import math
from cogdl.models import BaseModel, register_model


class BPRData(data.Dataset):
	def __init__(self, train_dict=None, num_item=0, num_ng=1, is_training=None, data_set_count=0, all_rating=None):
		super(BPRData, self).__init__()

		self.num_item = num_item
		self.train_dict = train_dict
		self.num_ng = num_ng
		self.is_training = is_training
		self.data_set_count = data_set_count
		self.all_rating = all_rating
		self.set_all_item = set(range(num_item))


	def ng_sample(self):
		# assert self.is_training, 'no need to sampling when testing'
		print('ng_sample----is----call-----')
		self.features_fill = []
		for user_id in self.train_dict:
			positive_list = self.train_dict[user_id]  # self.train_dict[user_id]
			all_positive_list = self.all_rating[user_id]
			# item_i: positive item ,,item_j:negative item
			# temp_neg=list(self.set_all_item-all_positive_list)
			# random.shuffle(temp_neg)
			# count=0
			# for item_i in positive_list:
			#     for t in range(self.num_ng):
			#         self.features_fill.append([user_id,item_i,temp_neg[count]])
			#         count+=1
			for item_i in positive_list:
				for t in range(self.num_ng):
					item_j = np.random.randint(self.num_item)
					while item_j in all_positive_list:
						item_j = np.random.randint(self.num_item)
					self.features_fill.append([user_id, item_i, item_j])
					# print('features_fill: ', self.features_fill)
	def __len__(self):
		return self.num_ng * self.data_set_count  # return self.num_ng*len(self.train_dict)

	def __getitem__(self, idx):
		features = self.features_fill
		if len(features) > 0 and len(features[idx]) > 0:
			user = features[idx][0]
			item_i = features[idx][1]
			item_j = features[idx][2]
			return user, item_i, item_j


class resData(data.Dataset):
	def __init__(self, train_dict=None, batch_size=0, num_item=0, all_pos=None):
		super(resData, self).__init__()

		self.train_dict = train_dict
		self.batch_size = batch_size
		self.all_pos_train = all_pos

		self.features_fill = []
		for user_id in self.train_dict:
			self.features_fill.append(user_id)
		self.set_all = set(range(num_item))

	def __len__(self):
		return math.ceil(len(self.train_dict) * 1.0 / self.batch_size)  # 这里的self.data_set_count==batch_size

	def __getitem__(self, idx):

		user_test = []
		item_test = []
		split_test = []
		for i in range(self.batch_size):  # 这里的self.data_set_count==batch_size
			index_my = self.batch_size * idx + i
			if index_my == len(self.train_dict):
				break
			user = self.features_fill[index_my]
			item_i_list = list(self.train_dict[user])
			item_j_list = list(self.set_all - self.all_pos_train[user])
			# pdb.set_trace()
			u_i = [user] * (len(item_i_list) + len(item_j_list))
			user_test.extend(u_i)
			item_test.extend(item_i_list)
			item_test.extend(item_j_list)
			split_test.append([(len(item_i_list) + len(item_j_list)), len(item_j_list)])

		# 实际上只用到一半去计算，不需要j的。
		return torch.from_numpy(np.array(user_test)), torch.from_numpy(np.array(item_test)), split_test


class LoadData(data.Dataset):
	def __init__(self, n_users, n_items, batch_size, interact_mat):
		self.n_users = n_users
		self.n_items = n_items
		self.batch_size = batch_size
		self.u_d = []
		self.i_d = []
		self.dropout = nn.Dropout(p=0.1)  # mess dropout
		self.n_hops = 3
		self.interact_mat = interact_mat
		super(LoadData, self).__init__()

	def readD(self, set_matrix, num_):
		user_d = []
		for i in range(num_):
			len_set = 1.0 / (len(set_matrix[i]) + 1)
			user_d.append(len_set)
		return user_d

	def readTrainSparseMatrix(self, set_matrix, is_user):
		user_items_matrix_i = []
		user_items_matrix_v = []
		if is_user:
			d_i = self.u_d
			d_j = self.i_d
		else:
			d_i = self.i_d
			d_j = self.u_d
		for i in set_matrix:
			len_set = len(set_matrix[i])
			for j in set_matrix[i]:
				user_items_matrix_i.append([i, j])
				d_i_j = np.sqrt(d_i[i] * d_j[j])
				# 1/sqrt((d_i+1)(d_j+1))
				user_items_matrix_v.append(d_i_j)  # (1./len_set)
		user_items_matrix_i = torch.cuda.LongTensor(user_items_matrix_i)
		user_items_matrix_v = torch.cuda.FloatTensor(user_items_matrix_v)
		return torch.sparse.FloatTensor(user_items_matrix_i.t(), user_items_matrix_v)

	def data(self):
		print(os.getcwd())
		dataset_base_path = os.getcwd() + '/../cogdl/data'
		training_user_set, training_item_set, training_set_count = np.load(
			dataset_base_path + '/datanpy/training_set.npy', allow_pickle=True)
		testing_user_set, testing_item_set, testing_set_count = np.load(
			dataset_base_path + '/datanpy/testing_set.npy', allow_pickle=True)
		user_rating_set_all = np.load(
			dataset_base_path + '/datanpy/user_rating_set_all.npy', allow_pickle=True).item()
		train_dataset = BPRData(
			train_dict=training_user_set, num_item=self.n_items, num_ng=5, is_training=True, \
			data_set_count=training_set_count, all_rating=user_rating_set_all)
		train_loader = DataLoader(train_dataset,
								  batch_size=self.batch_size, shuffle=True, num_workers=2)

		testing_dataset = BPRData(
			train_dict=testing_user_set, num_item=self.n_items, num_ng=5, is_training=True, \
			data_set_count=testing_set_count, all_rating=user_rating_set_all)
		testing_loader = DataLoader(testing_dataset,
										 batch_size=self.batch_size, shuffle=False, num_workers=0)

		self.u_d = self.readD(training_user_set, self.n_users)
		self.i_d = self.readD(training_item_set, self.n_items)
		# 1/(d_i+1)
		d_i_train = self.u_d
		d_j_train = self.i_d

		sparse_u_i = self.readTrainSparseMatrix(training_user_set, True)
		sparse_i_u = self.readTrainSparseMatrix(training_item_set, False)

		return sparse_u_i, sparse_i_u, d_i_train, d_j_train, train_loader, testing_loader

	def _sparse_dropout(self, x, rate=0.5):
		noise_shape = x._nnz()

		random_tensor = rate
		random_tensor += torch.rand(noise_shape).to(x.device)
		dropout_mask = torch.floor(random_tensor).type(torch.bool)
		i = x._indices()
		v = x._values()

		i = i[:, dropout_mask]
		v = v[dropout_mask]

		out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
		return out * (1.0 / (1 - rate))

	def forward(self, user_embed, item_embed, mess_dropout=True, edge_dropout=True):
		# user_embed: [n_users, channel]
		# item_embed: [n_items, channel]

		# all_embed: [n_users+n_items, channel]
		all_embed = torch.cat([user_embed, item_embed], dim=0)
		agg_embed = all_embed
		embs = [all_embed]

		for hop in range(self.n_hops):
			interact_mat = (
				self._sparse_dropout(self.interact_mat, 0.5) if edge_dropout else self.interact_mat
			)

			agg_embed = torch.sparse.mm(interact_mat, agg_embed)
			if mess_dropout:
				agg_embed = self.dropout(agg_embed)
			# agg_embed = F.normalize(agg_embed)
			embs.append(agg_embed)
		embs = torch.stack(embs, dim=1)  # [n_entity, n_hops+1, emb_size]
		return embs[: self.n_users, :], embs[self.n_users:, :]

@register_model("lr_gccf")
class LRGCCF(BaseModel):
	@staticmethod
	def add_args(parser):
		parser.add_argument('--factor_num', type=int, default=64)
		parser.add_argument('--batch_size', type=int, default=2048)
		parser.add_argument('--top_k', type=int, default=20)
		parser.add_argument("--n_negs", type=int, default=64, help="number of candidate negative")
		parser.add_argument("--user_item_matrix")
		parser.add_argument("--item_user_matrix")
		parser.add_argument("--d_i_train")
		parser.add_argument("--d_j_train")
	@classmethod
	def build_model_from_args(cls, args):
		return cls(
			args.n_users,
			args.n_items,
			args.factor_num,
			args.batch_size,
			args.top_k,
			args.user_item_matrix,
			args.item_user_matrix,
			args.d_i_train,
			args.d_j_train,
			args.adj_mat,
		)

	def __init__(
			self,
			n_users,
			n_items,
			factor_num,
			batch_size,
			top_k,
			user_item_matrix,
			item_user_matrix,
			d_i_train,
			d_j_train,
			adj_mat
	):
		super(LRGCCF, self).__init__()

		"""
		user_num: number of users;
		item_num: number of items;
		factor_num: number of predictive factors.
		"""
		self.n_users = n_users
		self.n_items = n_items
		self.adj_mat = adj_mat
		self.factor_num = factor_num
		self.batch_size = batch_size
		self.top_k = top_k

		self.embed_user = nn.Embedding(self.n_users, factor_num)
		self.embed_item = nn.Embedding(self.n_items, factor_num)

		nn.init.normal_(self.embed_user.weight, std=0.01)
		nn.init.normal_(self.embed_item.weight, std=0.01)

		self.gcn = LoadData(
			self.n_users,
			self.n_items,
			self.batch_size,
			self.adj_mat
		)

		self.user_item_matrix, self.item_user_matrix, self.d_i_train, self.d_j_train, self.train_loader, self.test_loader = self.gcn.data()
		for i in range(len(self.d_i_train)):
			self.d_i_train[i] = [self.d_i_train[i]]
		for i in range(len(self.d_j_train)):
			self.d_j_train[i] = [self.d_j_train[i]]

		self.d_i_train = torch.cuda.FloatTensor(self.d_i_train)
		self.d_j_train = torch.cuda.FloatTensor(self.d_j_train)
		self.d_i_train = self.d_i_train.expand(-1, self.factor_num)
		self.d_j_train = self.d_j_train.expand(-1, self.factor_num)

		self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.adj_mat)

	def _convert_sp_mat_to_sp_tensor(self, X):
		coo = X.tocoo()
		i = torch.LongTensor([coo.row, coo.col])
		v = torch.from_numpy(coo.data).float()
		return torch.sparse.FloatTensor(i, v, coo.shape)

	def generate(self, split=True):
		user_gcn_emb, item_gcn_emb = self.gcn(self.embed_user, self.embed_item, edge_dropout=False, mess_dropout=False)
		user_gcn_emb, item_gcn_emb = self.pooling(user_gcn_emb), self.pooling(item_gcn_emb)
		if split:
			return user_gcn_emb, item_gcn_emb
		else:
			return torch.cat([user_gcn_emb, item_gcn_emb], dim=0)

	def forward(self, batch):

		self.train_loader.dataset.ng_sample()
		print('train data of ng_sample is end.')
		for user, item_i, item_j in self.train_loader:
			user = user.cuda()
			item_i = item_i.cuda()
			item_j = item_j.cuda()
			users_embedding = self.embed_user.weight
			items_embedding = self.embed_item.weight
			# print(self.user_item_matrix)
			# print(items_embedding)
			# print(self.d_i_train)
			gcn1_users_embedding = (torch.sparse.mm(self.user_item_matrix, items_embedding) + users_embedding.mul(
				self.d_i_train))  # *2. #+ users_embedding
			gcn1_items_embedding = (torch.sparse.mm(self.item_user_matrix, users_embedding) + items_embedding.mul(
				self.d_j_train))  # *2. #+ items_embedding

			gcn2_users_embedding = (
						torch.sparse.mm(self.user_item_matrix, gcn1_items_embedding) + gcn1_users_embedding.mul(
					self.d_i_train))  # *2. + users_embedding
			gcn2_items_embedding = (
						torch.sparse.mm(self.item_user_matrix, gcn1_users_embedding) + gcn1_items_embedding.mul(
					self.d_j_train))  # *2. + items_embedding

			gcn3_users_embedding = (
						torch.sparse.mm(self.user_item_matrix, gcn2_items_embedding) + gcn2_users_embedding.mul(
					self.d_i_train))  # *2. + gcn1_users_embedding
			gcn3_items_embedding = (
						torch.sparse.mm(self.item_user_matrix, gcn2_users_embedding) + gcn2_items_embedding.mul(
					self.d_j_train))  # *2. + gcn1_items_embedding

			gcn4_users_embedding = (
						torch.sparse.mm(self.user_item_matrix, gcn3_items_embedding) + gcn3_users_embedding.mul(
					self.d_i_train))  # *2. + gcn1_users_embedding
			gcn4_items_embedding = (
						torch.sparse.mm(self.item_user_matrix, gcn3_users_embedding) + gcn3_items_embedding.mul(
					self.d_j_train))  # *2. + gcn1_items_embedding

			gcn_users_embedding = torch.cat(
				(users_embedding, gcn1_users_embedding, gcn2_users_embedding, gcn3_users_embedding,
				 gcn4_users_embedding),
				-1)  # +gcn4_users_embedding
			gcn_items_embedding = torch.cat(
				(items_embedding, gcn1_items_embedding, gcn2_items_embedding, gcn3_items_embedding,
				 gcn4_items_embedding),
				-1)  # +gcn4_items_embedding#

			user = F.embedding(user, gcn_users_embedding)
			item_i = F.embedding(item_i, gcn_items_embedding)
			item_j = F.embedding(item_j, gcn_items_embedding)
			# # pdb.set_trace()
			prediction_i = (user * item_i).sum(dim=-1)
			prediction_j = (user * item_j).sum(dim=-1)
			# loss=-((rediction_i-prediction_j).sigmoid())**2#self.loss(prediction_i,prediction_j)#.sum()
			l2_regulization = 0.01 * (user ** 2 + item_i ** 2 + item_j ** 2).sum(dim=-1)
			# l2_regulization = 0.01*((gcn1_users_embedding**2).sum(dim=-1).mean()+(gcn1_items_embedding**2).sum(dim=-1).mean())

			loss2 = -((prediction_i - prediction_j).sigmoid().log().mean())
			# loss= loss2 + l2_regulization
			loss = -((prediction_i - prediction_j)).sigmoid().log().mean() + l2_regulization.mean()

			# pdb.set_trace()

			return loss, prediction_i, prediction_j
	# def forward(self, user, item_i, item_j):
	#
	# 	users_embedding = self.embed_user.weight
	# 	items_embedding = self.embed_item.weight
	# 	# print(self.user_item_matrix)
	# 	# print(items_embedding)
	# 	# print(self.d_i_train)
	# 	gcn1_users_embedding = (torch.sparse.mm(self.user_item_matrix, items_embedding) + users_embedding.mul(
	# 		self.d_i_train))  # *2. #+ users_embedding
	# 	gcn1_items_embedding = (torch.sparse.mm(self.item_user_matrix, users_embedding) + items_embedding.mul(
	# 		self.d_j_train))  # *2. #+ items_embedding
	#
	# 	gcn2_users_embedding = (torch.sparse.mm(self.user_item_matrix, gcn1_items_embedding) + gcn1_users_embedding.mul(
	# 		self.d_i_train))  # *2. + users_embedding
	# 	gcn2_items_embedding = (torch.sparse.mm(self.item_user_matrix, gcn1_users_embedding) + gcn1_items_embedding.mul(
	# 		self.d_j_train))  # *2. + items_embedding
	#
	# 	gcn3_users_embedding = (torch.sparse.mm(self.user_item_matrix, gcn2_items_embedding) + gcn2_users_embedding.mul(
	# 		self.d_i_train))  # *2. + gcn1_users_embedding
	# 	gcn3_items_embedding = (torch.sparse.mm(self.item_user_matrix, gcn2_users_embedding) + gcn2_items_embedding.mul(
	# 		self.d_j_train))  # *2. + gcn1_items_embedding
	#
	# 	gcn4_users_embedding = (torch.sparse.mm(self.user_item_matrix, gcn3_items_embedding) + gcn3_users_embedding.mul(
	# 		self.d_i_train))  # *2. + gcn1_users_embedding
	# 	gcn4_items_embedding = (torch.sparse.mm(self.item_user_matrix, gcn3_users_embedding) + gcn3_items_embedding.mul(
	# 		self.d_j_train))  # *2. + gcn1_items_embedding
	#
	# 	gcn_users_embedding = torch.cat(
	# 		(users_embedding, gcn1_users_embedding, gcn2_users_embedding, gcn3_users_embedding, gcn4_users_embedding),
	# 		-1)  # +gcn4_users_embedding
	# 	gcn_items_embedding = torch.cat(
	# 		(items_embedding, gcn1_items_embedding, gcn2_items_embedding, gcn3_items_embedding, gcn4_items_embedding),
	# 		-1)  # +gcn4_items_embedding#
	#
	# 	user = F.embedding(user, gcn_users_embedding)
	# 	item_i = F.embedding(item_i, gcn_items_embedding)
	# 	item_j = F.embedding(item_j, gcn_items_embedding)
	# 	# # pdb.set_trace()
	# 	prediction_i = (user * item_i).sum(dim=-1)
	# 	prediction_j = (user * item_j).sum(dim=-1)
	# 	# loss=-((rediction_i-prediction_j).sigmoid())**2#self.loss(prediction_i,prediction_j)#.sum()
	# 	l2_regulization = 0.01 * (user ** 2 + item_i ** 2 + item_j ** 2).sum(dim=-1)
	# 	# l2_regulization = 0.01*((gcn1_users_embedding**2).sum(dim=-1).mean()+(gcn1_items_embedding**2).sum(dim=-1).mean())
	#
	# 	loss2 = -((prediction_i - prediction_j).sigmoid().log().mean())
	# 	# loss= loss2 + l2_regulization
	# 	loss = -((prediction_i - prediction_j)).sigmoid().log().mean() + l2_regulization.mean()
	# 	# pdb.set_trace()
	# 	return prediction_i, prediction_j, loss, loss2
