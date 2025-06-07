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

def visualize_attention_on_image(image_tensor, attn_map, grid_size, save_path='overlay.png'):
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2

    # 1. Convert ảnh về numpy
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
    image_np = (image_np * 255).astype(np.uint8)

    # 2. Resize attention map lên ảnh gốc
    attn_map = attn_map.detach().cpu().numpy()
    attn_map_resized = F.interpolate(
        torch.tensor(attn_map)[None, None, :, :].float(),
        size=(image_np.shape[0], image_np.shape[1]),
        mode='bilinear',
        align_corners=False
    )[0, 0].numpy()

    # 3. Normalize lại: 0–1
    attn_map_resized = (attn_map_resized - attn_map_resized.min()) / (attn_map_resized.max() - attn_map_resized.min() + 1e-8)
    attn_map_resized = 1.0 - attn_map_resized 
    # 4. Convert sang heatmap (mặc định JET: xanh → đỏ)
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_map_resized), cv2.COLORMAP_JET)

    # 5. Overlay heatmap lên ảnh gốc
    overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

    # 6. Hiển thị
    plt.imshow(overlay)
    plt.axis('off')
    plt.title("Attention Map")
    plt.savefig(save_path)
    plt.show()
    
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
            # print("x.shape:", x.shape) # 1 shot: 5,16,25,3,84,84 ==== 5 shot: 5,20,25,3,84,84 [n_way, batch_size, ...., *image_size]
            self.params.n_aug_support_samples = self.transform_aug # n_aug_support_samples + 8 
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
                    # print(feat_sup.shape, feat_qry.shape) #torch.Size([625, 128, 10, 10]) torch.Size([1875, 128, 10, 10])
                    pred, w_local2 = self.softmax(feat_sup, support_y, feat_qry,)
                    
                    
                    # query_xs_reshaped = query_xs.view(self.n_way, self.params.n_queries, self.params.n_symmetry_aug, *query_xs.shape[1:])

                    #  # Ví dụ: class 0, query ảnh 0
                    # c, q = 0, 0
                    # query_attn = w_local2[c, q]  # shape: [m] với m = num_patches (e.g., 16)
                    # flat_attn = query_attn.view(-1)
                    # top_flat_idx = torch.argmax(flat_attn).item()
                    
                    # # Giải ngược index: index = s_idx * num_patch + p_idx
                    # num_patch = query_attn.shape[1]
                    # support_idx = top_flat_idx // num_patch
                    # patch_idx = top_flat_idx % num_patch
                    # print(f"Top support: {support_idx}, Top patch: {patch_idx}")
                    
                    # attn_weights = query_attn[support_idx].detach().cpu()  # [num_patch] → [16]
                    # print(attn_weights.shape)
                    # attn_map = attn_weights.reshape(4, 4)  # nếu bạn dùng grid 4x4 = 16 patch
                    # query_patch_img = query_xs_reshaped[c, q, patch_idx + 9]  # patch_idx từ 0
                    # visualize_attention_on_image(query_patch_img, attn_map, grid_size=4, save_path=f'grad/{i}_overlay.png')
                    
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
            # print(proto_moving.shape) # [n_way, 8256]
            batch_size = self.params.sfc_bs
            alpha = self.params.alpha
            tempe = self.params.sim_temperature
            loss_ce_fn = nn.CrossEntropyLoss()
            iter_num = 100
            walk_times = 24

            for i in range(iter_num):
                # print(i, proto_moving.shape)
                #Lấy mean của 5 shot
                if proto_moving.dim() == 3: 
                    proto_moving = proto_moving.mean(dim=1)
                    
                weight = compute_weight_local(proto_moving, feat_sl, feat_sl, self.params.measure)
                # print(proto_moving.shape)
                idx_walk = torch.randperm(feat_sl.shape[2])[:walk_times]  # [n_aug]
                w_local = F.softmax(weight[:, :, :, idx_walk] * tempe, dim=-1)
                # print("feat_sl.shape",feat_sl.shape)
                feat_s = torch.sum(
                    feat_sl[:, :, idx_walk, :].unsqueeze(-3) * w_local.detach().unsqueeze(-1), dim=-2
                )  # [n_way, n_shot, dim]
                # print(i)
                # print("feat_s.shape", feat_s.shape)
                # print("feat_g.shape", feat_g.shape)
                support_x = alpha * feat_g + (1 - alpha) * feat_s
                # print("support_x.shape",support_x.shape)
                # print("support_x.mean(dim=1).shape", support_x.mean(dim=1).shape)
                proto_update = support_x.mean(dim=1).squeeze(1)   # [n_way, dim]
                # print(f'proto_update: {proto_update.shape}') #torch.Size([5, 5, 8256])   
                # print("proto_moving before update",proto_moving.shape)
                proto_moving = 0.9 * proto_moving + 0.1 * proto_update
                # print("[CHECK] proto_moving.shape after update:", proto_moving.shape) #torch.Size([5, 5, 8256])  
                spt_norm = torch.norm(support_x, p=2, dim=-1).unsqueeze(-1)
                support_x = support_x.div(spt_norm + 1e-6)

                # Train classifier
                SFC.train()
                sample_idxs = torch.randperm(self.n_way * self.n_shot)
                for j in range((len(sample_idxs) + batch_size - 1) // batch_size):
                    idxs = sample_idxs[j * batch_size:min((j + 1) * batch_size, len(sample_idxs))]
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
        # print(support_z.shape) #torch.Size([625, 128, 10, 10])
        support_z = self.dcov(support_z)
        # print(support_z.shape) #torch.Size([625, 8256])
        # print(query_z.shape) #torch.Size([1875, 128, 10, 10])
        query_z = self.dcov(query_z)
        # print(query_z.shape) #torch.Size([1875, 8256])
        
        support_ys = support_ys.view(self.n_way * self.n_shot, self.params.n_aug_support_samples, -1)
        global_ys = support_ys[:, 0, :].view(self.n_way, self.n_shot, -1)

        support_z = support_z.view(self.n_way, self.n_shot, self.params.n_aug_support_samples, -1)
        query_z = query_z.view(self.n_way, self.params.n_queries, self.params.n_aug_support_samples, -1)
        # print(support_z.shape, query_z.shape) #torch.Size([5, 5, 25, 8256]) torch.Size([5, 15, 25, 8256])
        feat_q = query_z[:, :, 0]                # original query #5, 15, 8256
        feat_ql_04 = query_z[:, :, 1:9]          # crop 0.4
        feat_ql_02 = query_z[:, :, 9:25]         # crop 0.2

        feat_g = support_z[:, :, 0]              # original support 5, 5, 8256
        feat_sl_04 = support_z[:, :, 1:9]
        feat_sl_02 = support_z[:, :, 9:25]

        # Define classifier
        SFC = nn.Linear(self.dim, self.params.n_way).cuda()
        SFC.bias.data.fill_(0)
        SFC.optimizer = torch.optim.AdamW(SFC.parameters(), lr=0.001, weight_decay=self.params.wd_test, eps=1e-4)

        # -------- PHASE 1: fusion with crop 0.4 --------
        #Lấy global làm  seed
        proto_seed_04 = feat_g.mean(dim=1).contiguous()  # [5, 8256]

        #For 1 shot 
        if self.n_shot == 1:
            pseudo_feat_g_04 = proto_seed_04.unsqueeze(1).expand(-1, self.n_shot, -1)  # [5, 1, 8256] - for 5 shot - [5,5,8256]
        elif self.n_shot == 5:
        #For 5 shot 
            pseudo_feat_g_04 = feat_g
        
        # Gọi đúng: không bao giờ dùng pseudo_feat_g_04 làm proto_seed
        proto_04 = featwalk_update(proto_seed_04, pseudo_feat_g_04, feat_sl_04, SFC, global_ys, self)
        if self.n_shot == 5:
            proto_04 = proto_04.mean(dim=1)
        # print("proto_seed_04:", proto_04.shape)     # [5, 8256]
        # print("pseudo_feat_g_04:", pseudo_feat_g_04.shape)  # [5, n_shot, 8256]
        # Query fusion for crop 0.4
        w_local_04 = compute_weight_local(proto_04, feat_ql_04, feat_sl_04, self.params.measure)
        w_local_04 = F.softmax(w_local_04 * tempe, dim=-1)
        feat_lq_04 = torch.sum(feat_ql_04.unsqueeze(-3) * w_local_04.unsqueeze(-1), dim=-2)
        query_x_04 = alpha * feat_q.unsqueeze(-2) + (1 - alpha) * feat_lq_04

        # -------- PHASE 2: fusion with crop 0.2 --------
        # print(query_x_04.shape) # For 5 shot 5,15,5,dim For 1 shot 5,15,1,dim
        proto_seed_02 = query_x_04.mean(dim=1) # For 1 shot 5,1,dim , For 5 shot 5,5,dim
        # Fix shape nếu còn thừa chiều size=1
        if proto_seed_02.dim() == 3:
            if self.n_shot == 1: 
                proto_seed_02 = proto_seed_02.squeeze(1)  # [5, 8256] 
            elif self.n_shot == 5: 
                proto_seed_02 = proto_seed_02.mean(dim=1) # [5, 8256]

        # Expand OK
        if self.n_shot == 1:
            pseudo_feat_g_02 = proto_seed_02.unsqueeze(1).expand(-1, self.n_shot, -1)  # [5, 1, 8256]
        elif self.n_shot == 5: 
            pseudo_feat_g_02 = proto_seed_02 # (5,5,8256)
            
        # print("proto_seed_02:", proto_seed_02.shape)           # phải là [5, 8256]
        # print("pseudo_feat_g_02:", pseudo_feat_g_02.shape)     # [5, n_shot, 8256]
        proto_02 = featwalk_update(proto_seed_02, pseudo_feat_g_02, feat_sl_02, SFC, global_ys, self)
        if self.n_shot == 5:
            proto_02 = proto_02.mean(dim=1)
        # Query fusion for crop 0.2
        w_local_02 = compute_weight_local(proto_02, feat_ql_02, feat_sl_02, self.params.measure)
        w_local_02 = F.softmax(w_local_02 * tempe, dim=-1)
        feat_lq_02 = torch.sum(feat_ql_02.unsqueeze(-3) * w_local_02.unsqueeze(-1), dim=-2)
        query_x_02 = alpha * feat_q.unsqueeze(-2) + (1 - alpha) * feat_lq_02
        query_x_02 = query_x_02 + query_x_04
        # Normalize final fused query
        query_x_02 = query_x_02 / (query_x_02.norm(p=2, dim=-1, keepdim=True) + 1e-6)

        with torch.no_grad():
            out = torch.sum(query_x_02 * SFC.weight, dim=-1) + SFC.bias  # [n_way, n_query]

        return out, w_local_02


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
