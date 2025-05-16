import copy
import torch
import torch.nn.functional as F
import networkx as nx
from models.line import LINE
import torch.nn as nn
import torch_clustering


class AdaptiveModule(nn.Module):
    def __init__(self):
        super(AdaptiveModule, self).__init__()
        self.register_buffer("a", torch.tensor(0.5))  # 初始化为0.5

    def update(self, loss_inter, loss_intra):
        if loss_inter > loss_intra + 1:
            self.a = torch.clamp(self.a - 0.1, 0, 1)
        else:
            self.a = torch.clamp(self.a + 0.1, 0, 1)

    def forward(self):
        return self.a



class RELIABLE(nn.Module):
    def __init__(self, n_views, n_samples, layer_dims, temperature, n_classes, drop_rate=0.5):
        super(RELIABLE, self).__init__()
        self.n_views = n_views
        self.n_classes = n_classes
        self.online_encoder = nn.ModuleList([FCN(layer_dims[i], drop_out=drop_rate) for i in range(n_views)])
        self.target_encoder = copy.deepcopy(self.online_encoder)

        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.cross_view_decoder = nn.ModuleList([MLP(layer_dims[i][-1], layer_dims[i][-1]) for i in range(n_views)])

        self.cl = ContrastiveLoss(temperature)
        self.feature_dim = [layer_dims[i][-1] for i in range(n_views)]

        self.weights = nn.Parameter(torch.full((self.n_views,), 1 / self.n_views), requires_grad=True)
        self.n_samples = n_samples
        self.psedo_labels = torch.zeros((self.n_samples,)).long().cuda()
        self.temperature_l = temperature
        self.temperature_f = temperature
        self.similarity = nn.CosineSimilarity(dim=2)

    @torch.no_grad()
    def compute_feature(self, data):
        z = [self.online_encoder[i](data[i]) for i in range(self.n_views)]
        p = [self.cross_view_decoder[i](z[i]) for i in range(self.n_views)]
        return z, p

    def forward(self, data, momentum, warm_up):

        self._update_target_branch(momentum)

        z = [self.online_encoder[i](data[i]) for i in range(self.n_views)]
        p = [self.cross_view_decoder[i](z[i]) for i in range(self.n_views)]

        z_t = [self.target_encoder[i](data[i]) for i in range(self.n_views)]

        if warm_up:
            mp = torch.eye(z[0].shape[0]).cuda()
            mp = [mp, mp]
        else:
            mp = [self.kernel_affinity(z_t[i]) for i in range(self.n_views)]

        adaptive_module = AdaptiveModule()

        a = adaptive_module()
        l_inter = a*(self.cl(p[0], z_t[1], mp[1]) + self.cl(p[1], z_t[0], mp[0]))
        l_intra = (1-a)*(self.cl(z[0], z_t[0], mp[0]) + self.cl(z[1], z_t[1], mp[1]))
        loss = l_inter + l_intra
        adaptive_module.update(l_inter.item(), l_intra.item())

        return loss

    @torch.no_grad()
    def kernel_affinity(self, z, temperature=0.1):
        z = L2norm(z)
        G = (2 - 2 * (z @ z.t())).clamp(min=0.)
        G = torch.exp(-G / temperature)
        G = G / G.sum(dim=1, keepdim=True)
        G = G.cpu().numpy()
        G = nx.from_numpy_array(G)
        # Using second-order proximate
        model = LINE(G, embedding_size=1024, order='second')
        model.train(batch_size=1024, epochs=1, verbose=2)
        embeddings = model.get_embeddings()
        # send data to GPU
        G = torch.tensor(embeddings).cuda(0)
        G = (2 - 2 * (z @ z.t())).clamp(min=0.)
        G = torch.exp(-G / temperature)
        G = G / G.sum(dim=1, keepdim=True)
        alpha = 0.5
        G = torch.eye(G.shape[0]).cuda() * alpha + G * (1 - alpha)  # 将单位矩阵（自连接）与亲和力矩阵 G 进行加权融合, 通过自连接和邻居信息的平衡，确保在高阶游走过程中，不会完全丧失当前样本自身的信息。
        return G

    @torch.no_grad()
    def _update_target_branch(self, momentum):
        for i in range(self.n_views):
            for param_o, param_t in zip(self.online_encoder[i].parameters(), self.target_encoder[i].parameters()):
                param_t.data = param_t.data * momentum + param_o.data * (1 - momentum)

    @torch.no_grad()
    def extract_feature(self, data, mask):
        N = data[0].shape[0]
        z = [torch.zeros(N, self.feature_dim[i]).cuda() for i in range(self.n_views)]
        for i in range(self.n_views):
            z[i][mask[:, i]] = self.target_encoder[i](data[i][mask[:, i]])

        for i in range(self.n_views):
            z[i][~mask[:, i]] = self.cross_view_decoder[1 - i](z[1 - i][~mask[:, i]])

        z = [self.cross_view_decoder[i](z[i]) for i in range(self.n_views)]
        z = [L2norm(z[i]) for i in range(self.n_views)]

        return z

    @torch.no_grad()
    def get_weights(self):
        with torch.no_grad():
            # self.weights is a learnable, initialize to uniform distribution
            weights = torch.softmax(self.weights, dim=0)
            return weights

    @torch.no_grad()
    def fusion(self, zs):
        zs = [z.cuda() for z in zs]
        weights = self.get_weights().cuda()

        attn_weights = []
        for i in range(self.n_views):
            attn = torch.sum(zs[i] * weights[i], dim=1)
            attn_weights.append(attn)

        attn_weights = torch.stack(attn_weights)
        attn_weights = torch.softmax(attn_weights, dim=0)

        weighted_zs = []
        for i in range(self.n_views):
            weighted_z = zs[i] * attn_weights[i].unsqueeze(1)
            weighted_zs.append(weighted_z)

        common_z = torch.sum(torch.stack(weighted_zs), dim=0)
        return common_z

    @torch.no_grad()
    def compute_centers(self, x, psedo_labels):
        n_samples = x.size(0)
        if len(psedo_labels.size()) > 1:
            weight = psedo_labels.T
        else:
            weight = torch.zeros(self.n_classes, n_samples).to(x)
            weight[psedo_labels, torch.arange(n_samples)] = 1
    
        weight = self._adjust_weights(weight, psedo_labels)
        affinity_matrix = self._compute_affinity_matrix(x)
        weight = torch.mm(weight, affinity_matrix)  # 矩阵乘法
        centers = torch.mm(weight, x)
        centers = F.normalize(centers, dim=1)
        return centers
    
    @torch.no_grad()
    def _adjust_weights(self, weight, psedo_labels):
        class_counts = torch.bincount(psedo_labels, minlength=self.n_classes)
        class_weights = 1.0 / (class_counts + 1e-6)  # 避免除以零
        class_weights = class_weights / class_weights.sum()  # 归一化
        weight = weight * class_weights.view(-1, 1)
        return weight
    
    @torch.no_grad()
    def _compute_affinity_matrix(self, x):
        affinity_matrix = torch.matmul(x, x.T)
        affinity_matrix = F.normalize(affinity_matrix, p=1, dim=1)
        return affinity_matrix

    @torch.no_grad()
    def clustering(self, features):
        kwargs = {
            'metric': 'cosine',
            'distributed': False,
            'random_state': 0,
            'n_clusters': self.n_classes,
            'verbose': False
        }
        clustering_model = torch_clustering.PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)
        psedo_labels = clustering_model.fit_predict(features.to(dtype=torch.float64))

        return psedo_labels

    @torch.no_grad()
    def compute_cluster_loss(self, q_centers, k_centers, psedo_labels):
        loss_single = self.compute_single_view_cluster_loss( q_centers, k_centers, psedo_labels)
        loss_fuse = self.compute_fused_view_cluster_loss( q_centers, k_centers, psedo_labels)
        loss = loss_single + loss_fuse
        return loss

    @torch.no_grad()
    def compute_single_view_cluster_loss(self, q_centers, k_centers, psedo_labels):
        d_q = q_centers.mm(q_centers.T) / self.temperature_l
        d_k = (q_centers * k_centers).sum(dim=1) / self.temperature_l
        d_q = d_q.float()
        d_q[torch.arange(self.n_classes), torch.arange(self.n_classes)] = d_k

        zero_classes = torch.arange(self.n_classes).cuda()[
            torch.sum(F.one_hot(torch.unique(psedo_labels), self.n_classes), dim=0) == 0]
        mask = torch.zeros((self.n_classes, self.n_classes), dtype=torch.bool, device=d_q.device)
        mask[:, zero_classes] = 1
        d_q.masked_fill_(mask, -10)
        pos = d_q.diag(0)
        pos = torch.sigmoid(pos)
        loss = - pos
        loss[zero_classes] = 0.
        loss = loss.sum() / (self.n_classes - len(zero_classes))
        return loss

    @torch.no_grad()
    def compute_fused_view_cluster_loss(self, q_centers, k_centers, psedo_labels):
        d_q = q_centers.mm(q_centers.T) / self.temperature_l
        d_k = (q_centers * k_centers).sum(dim=1) / self.temperature_l
        d_q = d_q.float()
        d_q[torch.arange(self.n_classes), torch.arange(self.n_classes)] = d_k
        zero_classes = torch.arange(self.n_classes).cuda()[
            torch.sum(F.one_hot(torch.unique(psedo_labels), self.n_classes), dim=0) == 0]
        mask = torch.zeros((self.n_classes, self.n_classes), dtype=torch.bool, device=d_q.device)
        mask[:, zero_classes] = 1
        d_q.masked_fill_(mask, -10)
        pos = d_q.diag(0)
        mask = torch.ones((self.n_classes, self.n_classes))
        mask = mask.fill_diagonal_(0).bool()
        neg = d_q[mask].reshape(-1, self.n_classes - 1)
        pos = torch.sigmoid(pos)
        neg = torch.sigmoid(neg)
        loss = torch.logsumexp(torch.cat([pos.reshape(self.n_classes, 1), neg], dim=1), dim=1)
        loss[zero_classes] = 0.
        loss = loss.sum() / (self.n_classes - len(zero_classes))

        return loss

    @torch.no_grad()
    def feature_loss(self, zi, z, w, y_pse):
        cross_view_distance = self.similarity(zi.unsqueeze(1), z.unsqueeze(0)) / self.temperature_f
        N = z.size(0)
        w = w + torch.eye(N, dtype=int).to(w.device)
        positive_loss = (w & y_pse) * cross_view_distance
        inter_view_distance = self.similarity(zi.unsqueeze(1), zi.unsqueeze(0)) / self.temperature_f
        positive_loss = -torch.sum(positive_loss)
        negated_w = w ^ True
        negated_y = y_pse ^ True
        SMALL_NUM = torch.log(torch.tensor(1e-45)).to(zi.device)
        negtive_cross = (negated_w & negated_y) * cross_view_distance
        negtive_cross[negtive_cross == 0.] = SMALL_NUM
        negtive_inter = (negated_w & negated_y) * inter_view_distance
        negtive_inter[negtive_inter == 0.] = SMALL_NUM
        negtive_similarity = torch.cat((negtive_inter, negtive_cross), dim=1) / self.temperature_f
        negtive_loss = torch.logsumexp(negtive_similarity, dim=1, keepdim=False)
        negtive_loss = torch.sum(negtive_loss)
        return (positive_loss + negtive_loss) / N


L2norm = nn.functional.normalize

class FCN(nn.Module):
    def __init__(self, dim_layer=None, norm_layer=None, act_layer=None, drop_out=0.0, norm_last_layer=True):
        super(FCN, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm1d
        layers = []
        for i in range(1, len(dim_layer) - 1):
            layers.append(nn.Linear(dim_layer[i - 1], dim_layer[i], bias=False))
            layers.append(norm_layer(dim_layer[i]))
            layers.append(act_layer())
            if drop_out != 0.0 and i != len(dim_layer) - 2:
                layers.append(nn.Dropout(drop_out))

        if norm_last_layer:
            layers.append(nn.Linear(dim_layer[-2], dim_layer[-1], bias=False))
            layers.append(nn.BatchNorm1d(dim_layer[-1], affine=False))
        else:
            layers.append(nn.Linear(dim_layer[-2], dim_layer[-1], bias=True))

        self.ffn = nn.Sequential(*layers)

    def forward(self, x):
        return self.ffn(x)

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out=None, hidden_ratio=4.0, act_layer=None):
        super(MLP, self).__init__()
        dim_out = dim_out or dim_in
        dim_hidden = int(dim_in * hidden_ratio)
        act_layer = act_layer or nn.ReLU
        self.mlp = nn.Sequential(nn.Linear(dim_in, dim_hidden),
                                 act_layer(),
                                 nn.Linear(dim_hidden, dim_out))

    def forward(self, x):
        x = self.mlp(x)
        return x

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, x_q, x_k, mask_pos=None):
        x_q = L2norm(x_q)
        x_k = L2norm(x_k)
        N = x_q.shape[0]
        if mask_pos is None:
            mask_pos = torch.eye(N).cuda()
        similarity = torch.div(torch.matmul(x_q, x_k.T), self.temperature)
        similarity = -torch.log(torch.softmax(similarity, dim=1))
        nll_loss = similarity * mask_pos / mask_pos.sum(dim=1, keepdim=True)
        loss = nll_loss.mean()
        return loss
