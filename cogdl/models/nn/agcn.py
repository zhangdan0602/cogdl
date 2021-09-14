# ! /usr/bin/python
# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         agcn
# Description:
# Author:       Zd
# Date:         2021/7/24
# -------------------------------------------------------------------------------
from torch.nn import Parameter

from cogdl.models import register_model, BaseModel
import torch.nn as nn
import torch


@register_model("agcn")
class AGCN(BaseModel):
	@staticmethod
	def add_args(parser):
		parser.add_argument('--free_emb_dim', type=int, default=64, help='embedding size')
		parser.add_argument('--gcn_layer', type=int, default=3, help='layer of gcn')
		parser.add_argument('--attr_union_dim', type=int, default=20, help='dim of attr union')
		parser.add_argument('--lambda1', type=float, default=0.001)
		parser.add_argument('--lambda2', type=float, default=0.001)

	@classmethod
	def build_model_from_args(cls, args):
		return cls(
			args,
		)

	def __init__(self, args):
		super(AGCN, self).__init__()

		self.n_users = args.n_users
		self.n_items = args.n_items
		self.adj_mat = args.adj_mat
		self.device= args.device
		self.free_emb_dim = args.free_emb_dim
		self.attr_union_dim = args.attr_union_dim
		self.gcn_layer = args.gcn_layer
		self.lambda1 = args.lambda1
		self.lambda2 = args.lambda2
		self.init_weights()
		self.graph_adj_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat)

	def init_weights(self):
		# 初始化user和item的free embedding
		self.user_emb = nn.init.normal_(torch.empty(self.n_users, self.free_emb_dim), mean=0, std=0.01)
		self.item_emb = nn.init.normal_(torch.empty(self.n_items, self.free_emb_dim), mean=0, std=0.01)
		self.user_emb = Parameter(self.user_emb)
		self.item_emb = Parameter(self.item_emb)

		# user属性初始化
		self.user_attrs_emb = nn.init.normal_(torch.empty(self.n_users, self.attr_union_dim), mean=0, std=0.01)
		self.user_attrs_emb = Parameter(self.user_attrs_emb)

		# item属性初始化
		self.item_attrs_emb = nn.init.normal_(torch.empty(self.n_items, self.attr_union_dim), mean=0, std=0.01)
		self.item_attrs_emb = Parameter(self.item_attrs_emb)

	def _convert_sp_mat_to_sp_tensor(self, X):
		coo = X.tocoo()
		i = torch.LongTensor([coo.row, coo.col])
		v = torch.from_numpy(coo.data).float()
		return torch.sparse.FloatTensor(i, v, coo.shape)

	def gcn_embedding(self):
		# 拼接 free embedding 和 attr embedding
		self.fused_user_emb = torch.cat((self.user_emb, self.user_attrs_emb), 1)
		self.fused_item_emb = torch.cat((self.item_emb, self.item_attrs_emb), 1)

		# 进入GCN
		self.feature_layer = torch.cat((self.fused_user_emb, self.fused_item_emb), 0)
		for layer in range(self.gcn_layer):
			neighbor_layer = torch.sparse.mm(self.graph_adj_mat.to(self.device), self.feature_layer)
			self.feature_layer = torch.add(neighbor_layer, self.feature_layer)

		self.final_user_emb, self.final_item_emb = torch.split(self.feature_layer,
															   [self.n_users, self.n_items],
															   0)
		return self.final_user_emb, self.final_item_emb

	def forward(self, batch):
		user_gcn_emb, item_gcn_emb = self.gcn_embedding()
		user = batch["users"]
		pos_item = batch["pos_items"]
		neg_item = batch["neg_items"]
		neg_item = neg_item.squeeze()
		u_gcn_embs = user_gcn_emb[user]
		pos_gcn_embs = item_gcn_emb[pos_item]
		neg_gcn_embs = item_gcn_emb[neg_item]
		loss, _, _ = self.create_bpr_loss(u_gcn_embs, pos_gcn_embs, neg_gcn_embs, user, pos_item, neg_item)
		return loss, _, _

	def create_bpr_loss(self, u_gcn_embs, pos_gcn_embs, neg_gcn_embs, user, pos_item, neg_item):
		ra_pos = torch.sum(torch.mul(u_gcn_embs, pos_gcn_embs), 1)
		ra_neg = torch.sum(torch.mul(u_gcn_embs, neg_gcn_embs), 1)
		with torch.no_grad():
			self.auc = torch.mean((ra_pos > ra_neg).float())
		bpr_loss = - torch.mean(torch.log(torch.clip(torch.sigmoid(ra_pos - ra_neg), 1e-10, 1.0)))

		# u_ego_embeddings = self.user_emb[user]
		# pos_ego_embeddings = self.item_emb[pos_item]
		# neg_ego_embeddings = self.item_emb[neg_item]
		regulation = self.lambda1 * torch.mean(torch.square(u_gcn_embs)) + \
					 self.lambda2 * torch.mean(torch.square(pos_gcn_embs) + torch.square(neg_gcn_embs))

		return bpr_loss + regulation, bpr_loss, regulation

	def generate(self):
		user_gcn_emb, item_gcn_emb = self.gcn_embedding()
		return user_gcn_emb, item_gcn_emb

	def rating(self, u_g_embeddings, pos_i_g_embeddings):
		return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())
