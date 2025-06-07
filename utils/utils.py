
import os
import shutil
import time
import pprint
import torch
import numpy as np
import os.path as osp
import random
import torch.nn.functional as F

def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_model(model, dir):
    model_dict = model.state_dict()
    file_dict = torch.load(dir)['state']
    for k, v in file_dict.items():
        if k not in model_dict:
            print(k)
    file_dict = {k: v for k, v in file_dict.items() if k in model_dict}
    model_dict.update(file_dict)
    model.load_state_dict(model_dict)
    return model

def compute_weight_local(feat_g,feat_ql,feat_sl,temperature=2.0):
    # feat_g : nk * dim
    # feat_l : nk * m * dim
    [_,k,m,dim] = feat_sl.shape
    [n,q,m,dim] = feat_ql.shape
    feat_g_expand = feat_g.unsqueeze(1).unsqueeze(2).expand(-1, feat_ql.size(1), feat_ql.size(2), -1)

    # feat_g_expand = feat_g.unsqueeze(2).expand_as(feat_ql)
    sim_gl = torch.cosine_similarity(feat_g_expand,feat_ql,dim=-1)
    I_opp_m = (1 - torch.eye(m)).unsqueeze(0).to(sim_gl.device)
    sim_gl = -(torch.matmul(sim_gl, I_opp_m).unsqueeze(-2))/(m-1)


    return sim_gl

#  proto_walk
# def compute_weight_local(feat_g, feat_ql, feat_sl, measure='cosine'):
#     # feat_g: [n_way, dim]
#     # feat_ql: [n_way, n_query, n_aug, dim]
#     # feat_sl: [n_way, n_shot, n_aug, dim]
#     assert feat_g.dim() == 2, f"feat_g must be [n_way, dim] but got {feat_g.shape}"
#     feat_g_expand = feat_g.unsqueeze(1).unsqueeze(2)  # [n_way, 1, 1, dim]  # [n_way, 1, 1, dim]
#     feat_g_expand = feat_g_expand.expand(-1, feat_ql.size(1), feat_ql.size(2), -1)

#     sim_gl = torch.cosine_similarity(feat_g_expand, feat_ql, dim=-1)  # [n_way, n_query, n_aug]
#     sim_ls = torch.einsum('nqmd,nkmd->nqkm', feat_ql, feat_sl)        # [n_way, n_query, n_shot, n_aug]
#     weight = sim_gl.unsqueeze(2) * sim_ls                             # [n_way, n_query, n_shot, n_aug]
#     return weight



def compute_weight_local(feat_g, feat_ql, feat_sl, measure='cosine'):
    # print("feat_g.shape",feat_g.shape)
    # print("feat_ql.shape",feat_ql.shape)
    # print("feat_sl.shape", feat_sl.shape)
    #Feat_g: prototype (mean of 5 shots or 1 shot)
    assert feat_g.dim() == 2, f"feat_g must be [n_way, dim] but got {feat_g.shape}"

    feat_g_expand = feat_g.unsqueeze(1).unsqueeze(2)  # [n_way, 1, 1, dim]
    feat_g_expand = feat_g_expand.expand(-1, feat_ql.size(1), feat_ql.size(2), -1)  # [n_way, n_query, n_aug, dim]

    sim_gl = torch.cosine_similarity(feat_g_expand, feat_ql, dim=-1)  # [n_way, n_query, n_aug]
    sim_ls = torch.einsum('nqmd,nkmd->nqkm', feat_ql, feat_sl)        # [n_way, n_query, n_shot, n_aug]

    weight = sim_gl.unsqueeze(2) * sim_ls  # [n_way, n_query, n_shot, n_aug]
    return weight


if __name__ == '__main__':
    feat_g = torch.randn((5,15,64))
    # feat_g = torch.ones((5,3,64))
    feat_sl = torch.randn((5,3,6,64))
    feat_ql = torch.randn((5,15,6,64))
    # feat_l = torch.ones((5,3,6,64))
    compute_weight_local(feat_g,feat_ql,feat_sl)
    # print(compute_weight_local(feat_g,feat_ql,feat_sl)[0,0])