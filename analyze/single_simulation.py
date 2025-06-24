import torch
from matplotlib import pyplot as plt
import torch.nn.functional as F
from tqdm import trange
import torch.nn as nn
from torch.optim import SGD
import os
import math
import argparse
from distillm.losses import *
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--tea-temp",default=5.0)
parser.add_argument("--stu-temp",default=2.0)
parser.add_argument("--final-tea-temp",default=2.0)
parser.add_argument("--final-stu-temp",default=1.0)
parser.add_argument("--decay-method",default="logarithmic_step_scheduler")
args = parser.parse_args()

data_num = 100
cls_num = 10000
lr = 1.0
# device = "cpu"
device = "cuda:0"
obj = "akl"
save_path = "/root/autodl-tmp/save/temp"
# save_path = f"../figures/{obj}/with_decoupled_temp_{args.tea_temp}_{args.stu_temp}"
# save_path = f"../figures/{obj}/with_nothing{args.tea_temp}_{args.stu_temp}"

# print(save_path)

if not os.path.exists(save_path):
    os.makedirs(save_path)

class Net(nn.Module):
    def __init__(self, data_num, cls_num, device, obj,tea_temp=2,stu_temp=2):
        super(Net, self).__init__()
        
        #  描述样本特征，每行对应一个样本的特征表示
        #  可以理解为学生视角下的样本表示，或者学生学到的特征
        self.h1 = nn.Parameter(torch.randn(data_num, 2, requires_grad=True, device=device))
        #  对初始值偏移操作
        self.h1.data = self.h1.data + 3
        #  表示 cls_num 个类别对应的二维嵌入向量。
        #  描述类别原型
        self.e1 = nn.Parameter(torch.randn(cls_num, 2, requires_grad=True, device=device))
        
        self.h2 = nn.Parameter(torch.randn(data_num, 2, requires_grad=True, device=device))
        self.e2 = nn.Parameter(torch.randn(cls_num, 2, requires_grad=True, device=device))
        self.h2.data = self.h1.data.clone()
        self.e2.data = self.e1.data.clone()

        self.h0 = torch.randn(data_num, 2)
        self.h0.data = self.h1.data.clone()
        # 教师向量
        self.h3 = torch.randn(data_num, 2, device=device) * 2
        self.e3 = torch.randn(cls_num, 2, device=device)

        self.tea_temp = tea_temp
        self.stu_temp = stu_temp
        self.obj = obj

    def cal_mse(self, a, b):
        mse = F.mse_loss(a, b, reduction="mean")
        return mse

    #  stu_logits, tea_logits
    def cal_kl(self, l1, l2):
        lprobs1 = torch.log_softmax(l1/self.tea_temp, -1)
        probs2 = torch.softmax(l2/self.stu_temp, -1)
        lprobs2 = torch.log_softmax(l2/self.stu_temp, -1)
        kl = (probs2 * (lprobs2 - lprobs1)).sum(-1)
        loss = kl.mean()
        return loss
    
    # def cal_kl(self, l1, l2):
    #     lprobs1 = torch.log_softmax(l1/self.stu_temp, -1)
    #     probs2 = torch.softmax(l2/self.stu_temp, -1)
    #     lprobs2 = torch.log_softmax(l2/self.stu_temp, -1)
    #     kl = (probs2 * (lprobs2 - lprobs1)).sum(-1)
    #     loss = kl.mean()
    #     return loss
    def cal_rkl(self, l1, l2):
        lprobs1 = torch.log_softmax(l1, -1)
        probs1 = torch.softmax(l1, -1)
        lprobs2 = torch.log_softmax(l2, -1)
        rkl = (probs1 * (lprobs1 - lprobs2)).sum(-1)
        loss = rkl.mean()
        return loss
    
    def cal_js(self, l1, l2):
        probs1 = torch.softmax(l1, -1)
        probs2 = torch.softmax(l2, -1)
        mprobs = (probs1 + probs2) / 2
        lprobs1 = torch.log(probs1 + 1e-9)
        lprobs2 = torch.log(probs2 + 1e-9)
        lmprobs = torch.log(mprobs + 1e-9)
        kl1 = probs1 * (lprobs1 - lmprobs)
        kl2 = probs2 * (lprobs2 - lmprobs)
        js = (kl1 + kl2) / 2
        loss = js.sum(-1).mean()
        return loss

    def cal_skl(self, l1, l2):
        probs1 = torch.softmax(l1, -1)
        probs2 = torch.softmax(l2, -1)
        probs1 = 0.9 * probs1 + 0.1 * probs2
        lprobs1 = torch.log(probs1 + 1e-9)
        lprobs2 = torch.log(probs2 + 1e-9)
        kl = probs2 * (lprobs2 - lprobs1)
        loss = kl.sum(-1).mean()
        return loss

    def cal_srkl(self, l1, l2):
        probs1 = torch.softmax(l1, -1)
        probs2 = torch.softmax(l2, -1)
        probs2 = 0.9 * probs2 + 0.1 * probs1
        lprobs1 = torch.log(probs1 + 1e-9)
        lprobs2 = torch.log(probs2 + 1e-9)
        kl = probs1 * (lprobs1 - lprobs2)
        loss = kl.sum(-1).mean()
        return loss

    def cal_akl(self, l1, l2):
        probs1 = torch.softmax(l1, -1)
        probs2 = torch.softmax(l2, -1)
        sorted_probs2, sorted_idx = probs2.sort(-1)
        sorted_probs1 = probs1.gather(-1, sorted_idx)
        gap = (sorted_probs2 - sorted_probs1).abs()
        cum_probs2 = torch.cumsum(sorted_probs2, -1)
        tail_mask = cum_probs2.lt(0.5).float()
        g_head = (gap * (1 - tail_mask)).sum(-1).detach()
        g_tail = (gap * tail_mask).sum(-1).detach()

        probs1 = torch.softmax(l1, -1)
        probs2 = torch.softmax(l2, -1)
        lprobs1 = torch.log_softmax(l1, -1)
        lprobs2 = torch.log_softmax(l2, -1)
        fkl = (probs2 * (lprobs2 - lprobs1)).sum(-1)
        rkl = (probs1 * (lprobs1 - lprobs2)).sum(-1)

        akl = (g_head / (g_head + g_tail)) * fkl + (g_tail / (g_head + g_tail)) * rkl
        loss = akl.mean()
        return loss
    
    # def temp_decay1(self, init_temp, step):
    #     decay_rate = 0.95  
    #     decay_steps = 1000  
      
    #     if step % decay_steps == 0 and step > 0:  
    #         new_temp = max(init_temp * (decay_rate ** (step // decay_steps)), self.min_temp)  
    #         return new_temp  
    #     return init_temp
    def temp_decay1(self,init_temp, final_temp, step, max_steps=10000, min_temp=0.1):
        if step is None or max_steps <= 0:
            return init_temp
        progress = min(step / max_steps, 1.0)
        return max(init_temp * (1 - progress) + final_temp * progress, min_temp)
    def temp_decay2(self, init_temp,final_temp, step,min_temp=0.5,k=4.0):
        decay_steps = 1000
        if step is not None or decay_steps <= 0:
            return init_temp
        t = min(step / decay_steps, 1.0)
        new_temp = init_temp * (final_temp / init_temp) ** (t ** k)
        return max(new_temp, min_temp)
    def logarithmic_step_scheduler(self,epoch, max_epochs):
        """对数阶梯调度：教师对数降温，学生阶梯调整"""
        # 教师温度：对数衰减 (初始高知识泛化，后期快速聚焦)
        T_t = np.exp(-1.5 * epoch/max_epochs) * 2.5 + 0.3
        T_t = max(T_t, 1.0)
        # 学生温度：阶梯式变化
        if epoch < max_epochs * 0.25:    # 第一阶段：基础学习
            T_s = 1.3
        elif epoch < max_epochs * 0.6:   # 第二阶段：增强探索
            T_s = 2.0
        else:                            # 第三阶段：精炼提升
            T_s = 0.6
        # T_s = max(T_s, 1.0)
        return T_t, T_s
    def temp_decay3(self,init_temp,final_temp, step,min_temp=0.5,k=4.0):
        decay_steps = 1000
        if step is None or step > decay_steps:
            return max(init_temp*math.exp(-k)+final_temp,min_temp)
        t = min(step / decay_steps, 1.0)
        new_temp = init_temp * math.exp(-k * t) + final_temp
        return max(new_temp, min_temp)
    def forward(self, share_head):
        if not share_head:
            stu_logits = self.h1.matmul(self.e1.transpose(-1, -2))
            tea_logits = self.h3.matmul(self.e3.transpose(-1, -2))
        else:
            stu_logits = self.h2.matmul(self.e2.transpose(-1, -2))
            #  使用e2.detach()而不是e3表示使用的相似头
            tea_logits = self.h3.matmul(self.e2.detach().transpose(-1, -2))

        if self.obj == "kl":
            loss = self.cal_kl(stu_logits, tea_logits)
        elif self.obj == "rkl":
            loss = self.cal_rkl(stu_logits, tea_logits)
        elif self.obj == "js":
            loss = self.cal_js(stu_logits, tea_logits)
        elif self.obj == "skl":
            loss = self.cal_skl(stu_logits, tea_logits)
        elif self.obj == "srkl":
            loss = self.cal_srkl(stu_logits, tea_logits)
        elif self.obj == "akl":
            loss = self.cal_akl(stu_logits, tea_logits)
        
        return loss

def compute_mean_cosine_similarity(student, teacher):
    """
    student: Tensor of shape [N, 2]
    teacher: Tensor of shape [N, 2]
    返回平均余弦相似度
    """
    return F.cosine_similarity(student, teacher, dim=1).mean().item()



for i in trange(1):
    
    model = Net(data_num, cls_num, device, obj,tea_temp=args.tea_temp,stu_temp=args.stu_temp)
    optim = SGD(model.parameters(), lr=lr, weight_decay=0.0)

    iters = 1000
    min_loss = 100
    kl_curve = []
    tea_temp_list = []
    stu_temp_list = []
    decay_method_name = args.decay_method
    decay_method = getattr(model, decay_method_name,None)
    if decay_method is None:
        raise ValueError("Invalid decay method")
    for i in range(iters):
        # model.tea_temp = decay_method(model.tea_temp, args.final_tea_temp, i)
        # model.stu_temp = decay_method(model.stu_temp, args.final_stu_temp, i)
        model.tea_temp = decay_method(i,1000)
        model.stu_temp = decay_method(i,1000)
        loss = model(False)
        loss.backward()
        optim.step()
        optim.zero_grad()
        
        kl_curve.append(loss.data)
        if loss < min_loss:
            plot_h1 = model.h1.data.clone()
            min_loss = loss
            tea_temp_list.append(model.tea_temp)
            stu_temp_list.append(model.stu_temp)
    kl_curve = torch.stack(kl_curve, 0)

    min_loss = 100
    for i in range(iters):
        model.tea_temp = decay_method(i,1000)
        model.stu_temp = decay_method(i,1000)
        loss = model(True)
        loss.backward()
        optim.step()
        optim.zero_grad()
        
        if loss < min_loss:
            plot_h2 = model.h2.data.clone()
            min_loss = loss
            tea_temp_list.append(model.tea_temp)
            stu_temp_list.append(model.stu_temp)
sim_before = compute_mean_cosine_similarity(model.h1, model.h3)
# print(f"KL 前平均余弦相似度: {sim_before:.4f}")
sim_after_diff_head =  compute_mean_cosine_similarity(plot_h1, model.h3)
# print(f"KL 后（不共享头）平均余弦相似度: {sim_after_diff_head:.4f}")
sim_after_share_head = compute_mean_cosine_similarity(plot_h2, model.h3)


with open(save_path+"/similarity.txt", "w") as f:
    f.write(f"KL 前平均余弦相似度: {sim_before:.4f}\n")
    f.write(f"KL 后（不共享头）平均余弦相似度: {sim_after_diff_head:.4f}\n")
    f.write(f"KL 后（共享头）平均余弦相似度: {sim_after_share_head:.4f}\n")

yhigh = int(max(model.h0[:, 1].max().item(), plot_h1[:, 1].max().item(), plot_h2[:, 1].max().item(), model.h3[:, 1].max().item())) + 1.5
ylow = int(min(model.h0[:, 1].min().item(), plot_h1[:, 1].min().item(), plot_h2[:, 1].min().item(), model.h3[:, 1].min().item())) - 1.5
xhigh = int(max(model.h0[:, 0].max().item(), plot_h1[:, 0].max().item(), plot_h2[:, 0].max().item(), model.h3[:, 0].max().item())) + 1.5
xlow = int(min(model.h0[:, 0].min().item(), plot_h1[:, 0].min().item(), plot_h2[:, 0].min().item(), model.h3[:, 0].min().item())) - 1.5

plt.xlim(xlow, xhigh)
plt.ylim(ylow, yhigh)


#  model.h0 是一个张量，代表 "Student" 类的嵌入向量。
#  [:, 0] 表示选取所有样本的第一个维度（即 x 坐标）
#  .detach防止梯度传播
plt.scatter(model.h0[:, 0].detach().cpu().numpy(), model.h0[:, 1].detach().cpu().numpy(), marker="^", c="r", label="Student")
plt.scatter(model.h3[:, 0].detach().cpu().numpy(), model.h3[:, 1].detach().cpu().numpy(), marker="*", c="b", label="Teacher")
plt.legend()
# plt.savefig(save_path + "before.eps")
# plt.savefig(save_path + "before.pdf")
plt.savefig(save_path + "/before.png")

plt.cla()
plt.xlim(xlow, xhigh)
plt.ylim(ylow, yhigh)
plt.scatter(plot_h1[:, 0].detach().cpu().numpy(), plot_h1[:, 1].detach().cpu().numpy(), marker="^", c="r", label="Student")
plt.scatter(model.h3[:, 0].detach().cpu().numpy(), model.h3[:, 1].detach().cpu().numpy(), marker="*", c="b", label="Teacher")
plt.legend()
# plt.savefig(save_path + "after_diff_head.eps")
# plt.savefig(save_path + "after_diff_head.pdf")
plt.savefig(save_path + "/after_diff_head.png")

plt.cla()
plt.xlim(xlow, xhigh)
plt.ylim(ylow, yhigh)
plt.scatter(plot_h2[:, 0].detach().cpu().numpy(), plot_h2[:, 1].detach().cpu().numpy(), marker="^", c="r", label="Student")
plt.scatter(model.h3[:, 0].detach().cpu().numpy(), model.h3[:, 1].detach().cpu().numpy(), marker="*", c="b", label="Teacher")
plt.legend()
# plt.savefig(save_path + "after_share_head.eps")
# plt.savefig(save_path + "after_share_head.pdf")
plt.savefig(save_path + "/after_share_head.png")

plt.figure(figsize=(10, 5))
plt.plot(tea_temp_list, label='Teacher Temperature', color='blue')
plt.plot(stu_temp_list, label='Student Temperature', color='red')
plt.xlabel('Iteration')
plt.ylabel('Temperature')
plt.title('Temperature Decay During Training')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_path, "temperature_decay.png"))
plt.close()