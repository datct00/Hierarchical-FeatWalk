import os
import sys
import time

import tqdm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from methods.bdc_module import BDC
import torch.nn.functional as F

sys.path.append("..")
import scipy
from scipy.stats import t
import network.resnet as resnet
from utils.loss import *
from sklearn.linear_model import LogisticRegression as LR
from utils.loss import DistillKL
from utils.utils import *
import math
from torch.nn.utils.weight_norm import WeightNorm

import warnings
warnings.filterwarnings("ignore")


def mean_confidence_interval(data, confidence=0.95,multi = 1):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m * multi, h * multi

def normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    out = x.div(norm)
    return out

def random_sample(linspace, max_idx, num_sample=5):
    sample_idx = np.random.choice(range(linspace), num_sample)
    sample_idx += np.sort(random.sample(list(range(0, max_idx, linspace)),num_sample))
    return sample_idx

def Triuvec(x,no_diag = False):
    batchSize, dim, dim = x.shape
    r = x.reshape(batchSize, dim * dim)
    I = torch.ones(dim, dim).triu()
    if no_diag:
        I -= torch.eye(dim,dim)
    I  = I.reshape(dim * dim)
    index = I.nonzero(as_tuple = False)
    # y = torch.zeros(batchSize, int(dim * (dim + 1) / 2), device=x.device).type(x.dtype)
    y = r[:, index].squeeze()
    return y

def Triumap(x,no_diag = False):

    batchSize, dim, dim, h, w = x.shape
    r = x.reshape(batchSize, dim * dim, h, w)
    I = torch.ones(dim, dim).triu()
    if no_diag:
        I -= torch.eye(dim,dim)
    I  = I.reshape(dim * dim)
    index = I.nonzero(as_tuple = False)
    # y = torch.zeros(batchSize, int(dim * (dim + 1) / 2), device=x.device).type(x.dtype)
    y = r[:, index, :, :].squeeze()
    return y

def Diagvec(x):
    batchSize, dim, dim = x.shape
    r = x.reshape(batchSize, dim * dim)
    I = torch.eye(dim, dim).triu().reshape(dim * dim)
    index = I.nonzero(as_tuple = False)
    y = r[:, index].squeeze()
    return y

class FeatWalk_Net(nn.Module):
    def __init__(self,params,num_classes = 5,):
        super(FeatWalk_Net, self).__init__()

        self.params = params

        if params.model == 'resnet12':
            self.feature = resnet.ResNet12(avg_pool=True,num_classes=64)
            resnet_layer_dim = [64, 160, 320, 640]
        elif params.model == 'resnet18':
            self.feature = resnet.ResNet18()
            resnet_layer_dim = [64, 128, 256, 512]

        self.resnet_layer_dim = resnet_layer_dim
        self.reduce_dim = params.reduce_dim
        self.feat_dim = self.feature.feat_dim
        self.dim = int(self.reduce_dim * (self.reduce_dim+1)/2)
        if resnet_layer_dim[-1] != self.reduce_dim:

            self.Conv = nn.Sequential(
                nn.Conv2d(resnet_layer_dim[-1], self.reduce_dim, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.reduce_dim),
                nn.ReLU(inplace=True)
            )
            self._init_weight(self.Conv.modules())

        drop_rate = params.drop_rate
        if self.params.embeding_way in ['BDC']:
            self.SFC = nn.Linear(self.dim, num_classes)
            self.SFC.bias.data.fill_(0)
        elif self.params.embeding_way in ['baseline++']:
            self.SFC = nn.Linear(self.reduce_dim, num_classes, bias=False)
            WeightNorm.apply(self.SFC, 'weight', dim=0)
        else:
            self.SFC = nn.Linear(self.reduce_dim, num_classes)

        self.drop = nn.Dropout(drop_rate)

        self.temperature = nn.Parameter(torch.log((1. /(2 * self.feat_dim[1] * self.feat_dim[2])* torch.ones(1, 1))),
                                            requires_grad=True)

        self.dcov = BDC(is_vec=True, input_dim=[self.reduce_dim,self.feature.feat_dim[1],self.feature.feat_dim[2]], dimension_reduction=self.reduce_dim)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if resnet_layer_dim[-1] != self.reduce_dim:
            self.dcov.conv_dr_block = self.Conv

        self.n_shot = params.n_shot
        self.n_way = params.n_way
        self.transform_aug = params.n_aug_support_samples + 8

    def _init_weight(self,modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def normalize(self,x):
        x = (x - torch.mean(x, dim=1).unsqueeze(1))
        return x

    def forward_feature(self, x):
        feat_map = self.feature(x, )
        if self.resnet_layer_dim[-1] != self.reduce_dim:
            feat_map = self.Conv(feat_map)
        out = feat_map
        return out

    def normalize_feature(self, x):
        if self.params.norm == 'center':
            x = x - x.mean(2).unsqueeze(2)
            return x
        else:
            return x

    def forward_pretrain(self, x):
        x = self.forward_feature(x)
        x = self.drop(x)
        return self.SFC(x)
    
    def train_loop(self,epoch,train_loader,optimizer):
        print_step = 100
        avg_loss = 0
        total_correct = 0
        iter_num = len(train_loader)
        total = 0
        loss_ce_fn = nn.CrossEntropyLoss()
        for i ,data in enumerate(train_loader):
            image , label = data
            image = image.cuda()
            label = label.cuda()
            out = self.forward_pretrain(image)
            loss =  loss_ce_fn(out, label)
            avg_loss = avg_loss + loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, pred = torch.max(out, 1)
            correct = (pred == label).sum().item()
            total_correct += correct
            total += label.size(0)
            if i % print_step == 0:
                print('\rEpoch {:d} | Batch: {:d}/{:d} | Loss: {:.4f} | Acc_train: {:.2f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1),correct/label.shape[0]*100), end=' ')
        print()

        return avg_loss / iter_num, float(total_correct) / total * 100

    def meta_val_loop(self,val_loader):
        acc = []
        for i, data in enumerate(val_loader):

            support_xs, support_ys, query_xs, query_ys = data
            support_xs = support_xs.cuda()
            query_xs = query_xs.cuda()
            split_size = 128
            if support_xs.squeeze(0).shape[0] >= split_size:
                feat_sup_ = []
                for j in range(math.ceil(support_xs.squeeze(0).shape[0] / split_size)):
                    fest_sup_item = self.forward_feature(
                        support_xs.squeeze(0)[j * split_size:min((j + 1) * split_size, support_xs.shape[1]), :, :, :],)
                    feat_sup_.append(fest_sup_item if len(fest_sup_item.shape) >= 1 else fest_sup_item.unsqueeze(0))
                feat_sup = torch.cat(feat_sup_, dim=0)
            else:
                feat_sup = self.forward_feature(support_xs.squeeze(0),)
            if query_xs.squeeze(0).shape[0] > split_size:
                feat_qry_ = []
                for j in range(math.ceil(query_xs.squeeze(0).shape[0] / split_size)):
                    feat_qry_item = self.forward_feature(
                        query_xs.squeeze(0)[j * split_size:min((j + 1) * split_size, query_xs.shape[1]), :, :, :],
                       )
                    feat_qry_.append(feat_qry_item if len(feat_qry_item.shape) > 1 else feat_qry_item.unsqueeze(0))

                feat_qry = torch.cat(feat_qry_, dim=0)
            else:
                feat_qry = self.forward_feature(query_xs.squeeze(0),)
            if self.params.LR:
                pred = self.LR(feat_sup, support_ys, feat_qry, query_ys)
            else:
                with torch.enable_grad():
                    pred = self.softmax(feat_sup, support_ys, feat_qry, )
                    _, pred = torch.max(pred, dim=-1)
            if self.params.n_symmetry_aug > 1:
                query_ys = query_ys.view(-1, self.params.n_symmetry_aug)
                query_ys = torch.mode(query_ys, dim=-1)[0]
            acc_epo = np.mean(pred.cpu().numpy() == query_ys.numpy())
            acc.append(acc_epo)
        return mean_confidence_interval(acc)

    def meta_test_loop(self,test_loader):
        acc = []
        for i, (x, _) in enumerate(test_loader):
            # print("x.shape:", x.shape)
            self.params.n_aug_support_samples = self.transform_aug
            self.params.n_symmetry_aug = 25
            # print(self.params.n_symmetry_aug)
            tic = time.time()
            x = x.contiguous().view(self.n_way, (self.n_shot + self.params.n_queries), *x.size()[2:])
            support_xs = x[:, :self.n_shot]  # [n_way, n_shot, n_aug, C, H, W]
            query_xs = x[:, self.n_shot:, :self.params.n_symmetry_aug]  # shape: [n_way, n_query, n_aug, C, H, W]
            
            support_xs = support_xs.contiguous().view(
                self.n_way * self.n_shot * self.params.n_aug_support_samples,
                x.size(3), x.size(4), x.size(5)
            ).cuda()

            query_xs = query_xs.contiguous().view(
                self.n_way * self.params.n_queries * self.params.n_symmetry_aug,
                x.size(3), x.size(4), x.size(5)
            ).cuda()
            # print("support_xs shape:", support_xs.shape)
            # print("query_xs shape:", query_xs.shape)
            # print("expecting:", self.n_way * self.params.n_queries * self.params.n_symmetry_aug, "×", x.size(3), x.size(4), x.size(5))

            support_y = torch.from_numpy(np.repeat(range(self.params.n_way),self.n_shot*self.params.n_aug_support_samples)).unsqueeze(0)
            split_size = 128
            if support_xs.shape[0] >= split_size:
                feat_sup_ = []
                for j in range(math.ceil(support_xs.shape[0]/split_size)):
                    fest_sup_item =self.forward_feature(support_xs[j*split_size:min((j+1)*split_size,support_xs.shape[0]),],)
                    feat_sup_.append(fest_sup_item if len(fest_sup_item.shape)>=1 else fest_sup_item.unsqueeze(0))
                feat_sup = torch.cat(feat_sup_,dim=0)
            else:
                feat_sup = self.forward_feature(support_xs)
            if query_xs.shape[0] >= split_size:
                feat_qry_ = []
                for j in range(math.ceil(query_xs.shape[0]/split_size)):
                    feat_qry_item = self.forward_feature(
                        query_xs[j * split_size:min((j + 1) * split_size, query_xs.shape[0]), ],)
                    feat_qry_.append(feat_qry_item if len(feat_qry_item.shape) > 1 else feat_qry_item.unsqueeze(0))

                feat_qry = torch.cat(feat_qry_,dim=0)
            else:
                feat_qry = self.forward_feature(query_xs,)

            if self.params.LR:
                pred = self.predict_wo_fc(feat_sup, support_y, feat_qry,)

            else:
                with torch.enable_grad():
                    pred = self.softmax(feat_sup, support_y, feat_qry,)
                    _,pred = torch.max(pred,dim=-1)

            query_ys = np.repeat(range(self.n_way), self.params.n_queries)
            pred = pred.view(-1)
            acc_epo = np.mean(pred.cpu().numpy() == query_ys)
            acc.append(acc_epo)
            print("\repisode {} acc: {:.2f} | avg_acc: {:.2f} +- {:.2f}, elapse : {:.2f}".format(i, acc_epo * 100,
                                                                                                 *mean_confidence_interval(
                                                                                                     acc, multi=100), (
                                                                                                             time.time() - tic) / 60),
                  end='')

        return mean_confidence_interval(acc)

    def distillation(self,epoch,train_loader,optimizer,model_t):
        print_step = 100
        avg_loss = 0
        total_correct = 0
        iter_num = len(train_loader)
        total = 0
        loss_div_fn = DistillKL(4)
        loss_ce_fn = nn.CrossEntropyLoss()
        for i, data in enumerate(train_loader):
            image, label = data
            image = image.cuda()
            label = label.cuda()
            with torch.no_grad():
                out_t = model_t.forward_pretrain(image)

            out= self.forward_pretrain(image)
            loss_ce = loss_ce_fn(out, label)
            loss_div = loss_div_fn(out, out_t)

            loss  = loss_ce * 0.5 + loss_div * 0.5
            avg_loss = avg_loss + loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, pred = torch.max(out, 1)
            correct = (pred == label).sum().item()
            total_correct += correct
            total += label.size(0)
            if i % print_step == 0:
                print('\rEpoch {:d} | Batch: {:d}/{:d} | Loss: {:.4f} | Acc_train: {:.2f}'.format(epoch, i,
                                                                                                  len(train_loader),
                                                                                                  avg_loss / float(
                                                                                                      i + 1),
                                                                                                  correct / label.shape[
                                                                                                      0] * 100),
                      end=' ')
        print()
        return avg_loss / iter_num, float(total_correct) / total * 100



    
    # new selective local fusion :
    def softmax(self, support_z, support_ys, query_z):
        support_ys = support_ys.cuda()
        loss_ce_fn = nn.CrossEntropyLoss()
        batch_size = self.params.sfc_bs
        walk_times = 24
        alpha = self.params.alpha
        tempe = self.params.sim_temperature
        def featwalk_update(proto_seed, feat_g, feat_sl, SFC, global_ys, self):
            proto_moving = proto_seed.clone()
            batch_size = self.params.sfc_bs
            alpha = self.params.alpha
            tempe = self.params.sim_temperature
            loss_ce_fn = nn.CrossEntropyLoss()
            iter_num = 100
            walk_times = 24
            num_sample = self.n_way * self.n_shot

            for _ in range(iter_num):
                proto_moving_detached = proto_moving.detach()
                print(proto_moving_detached.shape)
                weight = compute_weight_local(proto_moving_detached, feat_sl, feat_sl, self.params.measure)

                idx_walk = torch.randperm(feat_sl.shape[2])[:walk_times]
                w_local = F.softmax(weight[:, :, :, idx_walk] * tempe, dim=-1)

                feat_s = torch.sum(
                    feat_sl[:, :, idx_walk, :].unsqueeze(-3) * w_local.unsqueeze(-1), dim=-2
                )  # [n_way, n_shot, dim]

                support_x = alpha * feat_g + (1 - alpha) * feat_s
                support_x = support_x / (support_x.norm(p=2, dim=-1, keepdim=True) + 1e-6)

                proto_update = support_x.mean(dim=1)
                if proto_update.dim() == 3:
                    proto_update = proto_update.squeeze(1)
                proto_moving = 0.9 * proto_moving + 0.1 * proto_update
                print(f'proto_update: {proto_update.shape}')
                print("[CHECK] proto_moving.shape after update:", proto_moving.shape)

                # classifier update
                SFC.train()
                sample_idxs = torch.randperm(num_sample)
                for j in range((num_sample + batch_size - 1) // batch_size):
                    idxs = sample_idxs[j * batch_size:min((j + 1) * batch_size, num_sample)]
                    x = support_x[idxs // self.n_shot, idxs % self.n_shot]
                    y = global_ys[idxs // self.n_shot, idxs % self.n_shot].view(-1)

                    x = self.drop(x)
                    out = torch.sum(x * SFC.weight, dim=-1) + SFC.bias
                    loss = loss_ce_fn(out, y.long())
                    SFC.zero_grad()
                    loss.backward(retain_graph=True)
                    SFC.optimizer.step()

                SFC.eval()

            return proto_moving

        # BDC embedding
        support_z = self.dcov(support_z)
        query_z = self.dcov(query_z)

        support_ys = support_ys.view(self.n_way * self.n_shot, self.params.n_aug_support_samples, -1)
        global_ys = support_ys[:, 0, :].view(self.n_way, self.n_shot, -1)

        support_z = support_z.view(self.n_way, self.n_shot, self.params.n_aug_support_samples, -1)
        query_z = query_z.view(self.n_way, self.params.n_queries, self.params.n_aug_support_samples, -1)

        feat_g = support_z[:, :, 0]            # [n_way, n_shot, dim]
        feat_sl_04 = support_z[:, :, 1:9]
        feat_sl_02 = support_z[:, :, 9:25]

        feat_q = query_z[:, :, 0]              # [n_way, n_query, dim]
        feat_ql_04 = query_z[:, :, 1:9]
        feat_ql_02 = query_z[:, :, 9:25]

        # Create classifier
        SFC = nn.Linear(self.dim, self.params.n_way).cuda()
        SFC.bias.data.fill_(0)
        SFC.optimizer = torch.optim.AdamW(SFC.parameters(), lr=0.001, weight_decay=self.params.wd_test, eps=1e-4)

        # Phase 1: crop 0.4 fusion
        proto_seed_04 = feat_g.mean(dim=1).contiguous()
        pseudo_feat_g_04 = proto_seed_04.unsqueeze(1).expand(-1, self.n_shot, -1)
        print("proto_seed_04 shape:", proto_seed_04.shape)     # ✅ [5, 8256]
        print("pseudo_feat_g_04 shape:", pseudo_feat_g_04.shape)  # ✅ [5, 5, 8256]
        proto_04 = featwalk_update(proto_seed_04, pseudo_feat_g_04, feat_sl_04, SFC, global_ys, self)

        w_local_04 = compute_weight_local(proto_04, feat_ql_04, feat_sl_04, self.params.measure)
        w_local_04 = F.softmax(w_local_04 * tempe, dim=-1)
        feat_lq_04 = torch.sum(feat_ql_04.unsqueeze(-3) * w_local_04.unsqueeze(-1), dim=-2)
        query_x_04 = alpha * feat_q.unsqueeze(-2) + (1 - alpha) * feat_lq_04

        # Phase 2: crop 0.2 fusion
        proto_seed_02 = query_x_04.mean(dim=1)
        if proto_seed_02.dim() == 3:
            proto_seed_02 = proto_seed_02.squeeze(1)
        pseudo_feat_g_02 = proto_seed_02.unsqueeze(1).expand(-1, self.n_shot, -1)
        proto_02 = featwalk_update(proto_seed_02, pseudo_feat_g_02, feat_sl_02, SFC, global_ys, self)

        w_local_02 = compute_weight_local(proto_02, feat_ql_02, feat_sl_02, self.params.measure)
        w_local_02 = F.softmax(w_local_02 * tempe, dim=-1)
        feat_lq_02 = torch.sum(feat_ql_02.unsqueeze(-3) * w_local_02.unsqueeze(-1), dim=-2)
        query_x_02 = alpha * feat_q.unsqueeze(-2) + (1 - alpha) * feat_lq_02
        query_x_02 = query_x_02 / (query_x_02.norm(p=2, dim=-1, keepdim=True) + 1e-6)

        with torch.no_grad():
            out = torch.sum(query_x_02 * SFC.weight, dim=-1) + SFC.bias  # [n_way, n_query]

        return out

    def LR(self,support_z,support_ys,query_z,query_ys):

        clf = LR(penalty='l2',
                 random_state=0,
                 C=self.params.penalty_c,
                 solver='lbfgs',
                 max_iter=1000,
                 multi_class='multinomial')

        spt_norm = torch.norm(support_z, p=2, dim=1).unsqueeze(1).expand_as(support_z)
        spt_normalized = support_z.div(spt_norm  + 1e-6)

        qry_norm = torch.norm(query_z, p=2, dim=1).unsqueeze(1).expand_as(query_z)
        qry_normalized = query_z.div(qry_norm + 1e-6)

        z_support = spt_normalized.detach().cpu().numpy()
        z_query = qry_normalized.detach().cpu().numpy()

        y_support = np.repeat(range(self.params.n_way), self.n_shot)

        clf.fit(z_support, y_support)

        return torch.from_numpy(clf.predict(z_query))
