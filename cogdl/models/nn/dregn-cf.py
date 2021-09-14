# ! /usr/bin/python
# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         dregn-cf
# Description:
# Author:       Zd
# Date:         2021/7/19
# -------------------------------------------------------------------------------
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from cogdl.models import BaseModel, register_model


@register_model("dregn-cf")
class DREGN_CF(BaseModel):
	@staticmethod
	def add_args(parser):
		parser.add_argument('--dim', type=int, default=64, help='embedding size')
		parser.add_argument('--n_layers', type=int, default=3, help='layer size')
		parser.add_argument('--keep_prob', type=float, default=1.0)
		parser.add_argument('--a_split', type=bool, default=False)
		parser.add_argument('--dropout', type=bool, default=False)
		parser.add_argument('--dr_upper_bound', type=float, default=70.0)

	@classmethod
	def build_model_from_args(cls, args):
		return cls(
			args,
		)

	def __init__(self, args):
		super(DREGN_CF, self).__init__()

		self.n_users = args.n_users
		self.n_items = args.n_items
		self.adj_mat = args.adj_mat
		self.u_freqs = args.u_freqs
		self.p_u = torch.from_numpy(self.u_freqs / self.u_freqs.sum())
		self.ns = args.ns
		self.train_user_set = args.train_user_set

		self.emb_size = args.dim
		self.n_layers = args.n_layers
		self.keep_prob = args.keep_prob
		self.a_split = args.a_split
		self.dropout = args.dropout
		self.dr_upper_bound = args.dr_upper_bound

		self._init_weight()
		self.user_embed = nn.Parameter(self.user_embed)
		self.item_embed = nn.Parameter(self.item_embed)

		self.f = nn.Sigmoid()

	def _init_weight(self):
		initializer = nn.init.xavier_uniform_
		self.user_embed = initializer(torch.empty(self.n_users, self.emb_size))
		self.item_embed = initializer(torch.empty(self.n_items, self.emb_size))
		self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.adj_mat)

	def _convert_sp_mat_to_sp_tensor(self, X):
		coo = X.tocoo()
		i = torch.LongTensor([coo.row, coo.col])
		v = torch.from_numpy(coo.data).float()
		return torch.sparse.FloatTensor(i, v, coo.shape)

	def __dropout_x(self, x, keep_prob):
		size = x.size()
		index = x.indices().t()
		values = x.values()
		random_index = torch.rand(len(values)) + keep_prob
		random_index = random_index.int().bool()
		index = index[random_index]
		values = values[random_index] / keep_prob
		g = torch.sparse.FloatTensor(index.t(), values, size)
		return g

	def __dropout(self, keep_prob):
		if self.a_split:
			graph = []
			for g in self.sparse_norm_adj:
				graph.append(self.__dropout_x(g, keep_prob))
		else:
			graph = self.__dropout_x(self.sparse_norm_adj, keep_prob)
		return graph

	def gcn(self):
		users_emb = self.user_embed
		items_emb = self.item_embed
		all_emb = torch.cat([users_emb, items_emb])
		embs = [all_emb]
		if self.dropout:
			if self.training:
				g_droped = self.__dropout(self.keep_prob)
			else:
				g_droped = self.sparse_norm_adj
		else:
			g_droped = self.sparse_norm_adj

		for layer in range(self.n_layers):
			if self.a_split:
				temp_emb = []
				for f in range(len(g_droped)):
					temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
				side_emb = torch.cat(temp_emb, dim=0)
				all_emb = side_emb
			else:
				all_emb = torch.sparse.mm(g_droped.to(self.device), all_emb.to(self.device))
			embs.append(all_emb)
		embs = torch.stack(embs, dim=1)
		light_out = torch.mean(embs, dim=1)
		users, items = torch.split(
			light_out, [self.n_users, self.n_items]
		)
		return users, items

	def get_rankingDRE_loss_nois(self, r, pos_mask, pi_p_user):
		# Non-negative risk correction:  enable
		# Importance sampling estimator: disable
		# default setting
		# print(r.shape)
		# print(pos_mask.shape)
		# print(pi_p_user.shape)
		# weights and masks
		item_rank_w_neg = r.detach().clone()
		item_rank_w_pos = r.detach().clone().pow_(-1)
		# print(item_rank_w_pos.shape)
		# print(item_rank_w_neg.shape)
		item_w_nu_pos = item_rank_w_pos.mul(pos_mask.to(self.device))
		item_w_nu_neg = pos_mask.to(self.device).mul_(item_rank_w_neg)

		# term for non-negative risk correction
		r2 = r.pow(2)  # r.pow(2)
		C = 1 / self.dr_upper_bound
		corr = self.normalized_weighted_mean(r2, item_w_nu_pos).mul_(C * (1 / 2))

		# risk for samples from p(i|u,y=+1) (p_nu)
		term_a = self.normalized_weighted_mean(r, item_w_nu_pos).neg_()
		term_a_neg = self.normalized_weighted_mean(r2, item_w_nu_pos)
		term_a_neg = term_a_neg.sub_(self.normalized_weighted_mean(r2, item_w_nu_neg))
		term_a = term_a.add_(term_a_neg.mul_(pi_p_user))
		term_a = term_a.add_(corr)

		# risk for samples from p(i|u) (p_de)
		term_b = self.normalized_weighted_mean(r2, item_rank_w_neg).mul_((1 / 2))
		term_b = term_b.sub_(corr)
		base_loss = torch.mean(term_a + torch.clamp(term_b, min=0))

		return base_loss

	def normalized_weighted_mean(self, elements, weights):
		s = torch.sum(elements * weights, 1)
		return s.div_(torch.sum(weights, 1))
		# return s

	def rating(self, u_g_embeddings, pos_i_g_embeddings):
		return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

	def generate(self):
		user_gcn_emb, item_gcn_emb = self.gcn()
		return user_gcn_emb, item_gcn_emb

	def forward(self, batch=None):
		# print('start forwarding...')
		user = batch["users"]

		pos_item = batch["pos_items"]
		# print(pos_item)
		neg_item = batch["neg_items"]
		# users = batch["users1"]
		items = batch["items"]
		user_pos_inds = batch["user_pos_inds"]
		item_pos_inds = batch["item_pos_inds"]
		user_gcn_emb, item_gcn_emb = self.gcn()
		# print(user_gcn_emb[users])
		# print(user_gcn_emb[user])
		# print(item_gcn_emb[items])
		# print(item_gcn_emb[pos_item])
		# print(self.user_embed[users])
		# print(self.user_embed[user])
		# print(self.item_embed[items])
		# print(self.item_embed[pos_item])
		u_gcn_embs = user_gcn_emb[user]
		pos_gcn_embs = item_gcn_emb[items]
		u_ego_embeddings = self.user_embed[user]
		pos_ego_embeddings = self.item_embed[items]
		# u_gcn_embs = user_gcn_emb[user]
		# pos_gcn_embs = item_gcn_emb[pos_item]
		neg_gcn_embs = item_gcn_emb[neg_item]
		neg_ego_embeddings = self.item_embed[neg_item]
		# print(len(users))
		# print(len(items))
		train_users = []
		# u_inx = {}
		# for index, u in enumerate(user):
		# 	u = u.item()
		# 	train_users.append([[u] * len(self.train_user_set[u]), self.train_user_set[u]])
		# # 	u_inx[index] = u
		# train_users_data = np.concatenate(train_users, 1).T
		# train_users_data = train_users_data[:, 0]
		# user_freq = Counter(train_users_data)
		# cnt = 0
		# for (u, f) in user_freq.items():
		# 	cnt += f
		# u_prop = {}
		# for u, f in user_freq.items():
		# 	u_prop[u] = f / cnt
		# u_p = []
		# for i in users:
		# 	if i in list(user_freq.keys()):
		# 		u_p.append(u_prop[i.item()])
		# 	else:
		# 		u_p.append(0)

		# u_freqs = np.array(list(user_freq.values()))
		# u_freqs_prop = u_freqs / u_freqs.sum()

		# self.p_u = torch.from_numpy(np.array(u_p)).to(self.device)
		users = user.to(self.device, non_blocking=True)
		items = items.to(self.device, non_blocking=True)

		# uid_id = {}
		# for idx, u in enumerate(users.cpu()):
		# 	if u.item() not in uid_id:
		# 		uid_id[u.item()] = idx
		# user_pos_indexs = []
		# for i in user_pos_inds.numpy():
		# 	user_pos_indexs.append(uid_id[i.item()])
		# user_pos_inds = torch.from_numpy(np.array(user_pos_indexs))

		pi_p_user = self.p_u[users].to(self.device, non_blocking=True)
		# pi_p_user = self.p_u.to(self.device, non_blocking=True)
		# print('len of users: ', len(users))
		# print('len of items: ', len(items))

		# user_pos_inds1 = {}
		# user_pos_inds11 = []
		# cnt = 0
		# for i in user_pos_inds.numpy():
		# 	if i not in user_pos_inds1.keys():
		# 		user_pos_inds1[i] = cnt
		# 		cnt += 1
		# for k in user_pos_inds.numpy():
		# 	user_pos_inds11.append(user_pos_inds1[k])
		# user_pos_inds = torch.from_numpy(np.array(user_pos_inds11))
		# print(type(user_pos_inds), user_pos_inds.shape)
		# print(user_pos_inds)

		pos_mask = torch.cuda.FloatTensor(len(users), len(items)).fill_(0)
		pos_mask[user_pos_inds, item_pos_inds] = 1.0

		# print('start cal loss...')
		loss, base, reg = self.create_bpr_loss(users, items, pos_mask, pi_p_user, u_gcn_embs, pos_gcn_embs,
											   u_ego_embeddings, pos_ego_embeddings)
		# print("loss: ", loss, "\tbase: ", base, "\treg: ", reg)
		return loss, base, reg

	def create_bpr_loss(self,
						users,
						items,
						pos_mask,
						pi_p_user,
						u_gcn_embs,
						pos_gcn_embs,
						u_ego_embeddings,
						pos_ego_embeddings):

		# print(u_gcn_embs.shape)
		# print(pos_gcn_embs.shape)
		# print(pos_gcn_embs.t().shape)
		# construct estimated desnity ratio
		r = torch.matmul(u_gcn_embs, pos_gcn_embs.t())
		r = torch.nn.functional.softplus(r)
		base_loss = self.get_rankingDRE_loss_nois(r, pos_mask, pi_p_user)

		reg_loss = (1 / 2) * u_ego_embeddings.norm(2).pow(2).div(float(len(users)))
		reg_loss += (1 / 2) * pos_ego_embeddings.norm(2).pow(2).div(float(len(items)))

		return base_loss + reg_loss * 5e-2, base_loss, reg_loss
