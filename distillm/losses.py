import torch
import torch.nn.functional as F



def get_ratio(teacher_logits, logits, mu=0.5):
    # [B, L, V]
    # 将teacher_logits中的无穷值（inf或-inf）替换为0；
    teacher_logits = torch.masked_fill(teacher_logits, torch.isinf(teacher_logits), 0).to(torch.float32)
    logits = torch.masked_fill(logits, torch.isinf(logits), 0).to(torch.float32)
    # 输入句子为["The cat sat", "An apple is red"]
    # 对于 "cat" 这个位置，教师模型可能输出 [2.0, -1.0, 0.5, 0.1]，
    # #softmax 后变为 [0.6, 0.1, 0.2, 0.1]。
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32).detach()
    # 排序后可能是 [0.6, 0.2, 0.1, 0.1]，对应的索引是 [0, 2, 1, 3]
    re_teacher_probs, idx = teacher_probs.sort(dim=-1, descending=True)
    # 将学生模型的概率按照教师排序的位置重新排列。
    # 目的是为了对齐教师与学生模型在各个类别上的预测分布，便于后续计算误差。
    re_student_probs = student_probs.gather(dim=-1, index=idx)
    # 计算教师与学生模型在各类别上的概率差异；
    errors = torch.abs(re_teacher_probs - re_student_probs)
    # 对教师模型的概率沿类别维度累加，用于后续确定重要类别区域。
    cum_sum = torch.cumsum(re_teacher_probs, dim=-1) # B,L,V
    
    # 只针对重要区域，或者说误差小的区域
    # 例如
    mask = cum_sum > mu
    mask[:,:,0]=False #第一个概率一定要置False，对应第一个概率>0.5时mask全True
    #   ghead
    s1 = torch.masked_fill(errors, mask, 0.0).sum(dim=-1)
    #   gtail
    s2 = torch.masked_fill(errors, ~mask, 0.0).sum(dim=-1)


    return s1/(s1+s2), s2/(s1+s2)

# def get_kl(args,logits, teacher_logits, inf_mask, mask, ratio=None,mu=0.5):
#     tea_temp = args.tea_temp
#     stu_temp = args.stu_temp
#     #ratio: [B,L]
#     # p
#     teacher_probs = F.softmax(teacher_logits/tea_temp, dim=-1, dtype=torch.float32)
#     # log p
#     teacher_logprobs = F.log_softmax(teacher_logits/tea_temp, dim=-1, dtype=torch.float32)
#     # p*logp
#     teacher_prod_probs =  torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
#     # Σp*logp
#     teacher_x =  torch.sum(teacher_prod_probs, dim=-1).view(-1)
#     #logq
#     logprobs = F.log_softmax(logits/stu_temp, dim=-1, dtype=torch.float32)
#     #p*logq
#     prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
#     #Σp*logq
#     x = torch.sum(prod_probs, dim=-1).view(-1) # [B,L]->[BL]

#     if ratio == None:
#         distil_loss = torch.sum((teacher_x-x) * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
#     else:
#         distil_loss = torch.sum((teacher_x-x) * ratio.view(-1) * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
#     return distil_loss


def get_kl(args,logits, teacher_logits, inf_mask, mask, ratio=None,mu=0.5):
    tea_temp = args.tea_temp
    stu_temp = args.stu_temp
    #ratio: [B,L]
    # p
    teacher_probs = F.softmax(teacher_logits,dim=-1, dtype=torch.float32)
    # log p
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    # p*logp
    teacher_prod_probs =  torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
    # Σp*logp
    teacher_x =  torch.sum(teacher_prod_probs, dim=-1).view(-1)
    #logq
    logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    #p*logq
    prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
    #Σp*logq
    x = torch.sum(prod_probs, dim=-1).view(-1) # [B,L]->[BL]

    if ratio == None:
        distil_loss = torch.sum((teacher_x-x) * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    else:
        distil_loss = torch.sum((teacher_x-x) * ratio.view(-1) * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def AKL(args, logits,teacher_logits,no_model_batch):
    inf_mask = torch.isinf(logits) # [batch, seq, vocab]
    mask = (no_model_batch["label"] != -100).int() # [batch, seq]
    h_ratio, l_ratio = get_ratio(teacher_logits, logits)
    # distil_loss =get_kl(args,teacher_logits, logits, inf_mask, mask, h_ratio) + (args.stu_temp**2)*get_kl(args,logits,teacher_logits, inf_mask, mask, l_ratio)
    # distil_loss = get_kl(args,teacher_logits, logits, inf_mask, mask, h_ratio) + (args.stu_temp*args.tea_temp)*get_kl(args,logits,teacher_logits, inf_mask, mask, l_ratio)
    distil_loss = get_kl(args,teacher_logits, logits, inf_mask, mask, h_ratio) + get_kl(args,logits,teacher_logits, inf_mask, mask, l_ratio)
    return distil_loss

def decoupled_temp_kl(args,logits, teacher_logits, no_model_batch, lam=0.1,use_scaling=True):
    
    teacher_probs = F.softmax(teacher_logits/args.tea_temp,dim=-1,dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits/args.tea_temp,dim=-1,dtype=torch.float32)
    inf_mask = torch.isinf(logits)
    student_logprobs = F.log_softmax(logits/args.stu_temp,dim=-1,dtype=torch.float32)
    kld = teacher_probs*(teacher_logprobs-student_logprobs)
    x = torch.sum(kld, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    if use_scaling == True:
        return distil_loss ** args.tea_temp
    return distil_loss
# def forward_kl(args,logits, teacher_logits, no_model_batch):
#     tea_temp = args.tea_temp
#     teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
#     teacher_logprobs=F.log_softmax(teacher_logits,dim=-1,dtype=torch.float32)
#     inf_mask = torch.isinf(logits)
#     student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
#     # prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
#     prod_probs = torch.masked_fill(teacher_probs * (teacher_logprobs-student_logprobs), inf_mask, 0)
#     x = torch.sum(prod_probs, dim=-1).view(-1)
#     mask = (no_model_batch["label"] != -100).int()
#     distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
#     return distil_loss

def forward_kl(args,logits, teacher_logits, no_model_batch):
    args.tea_temp = 1
    # p
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(logits)
    #log q
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def reverse_kl(args,logits, teacher_logits, no_model_batch):
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)
    prod_probs = torch.masked_fill(student_probs * teacher_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def symmetric_kl(args,logits, teacher_logits, no_model_batch, lam=0.9):
    for_kl = forward_kl(logits, teacher_logits, no_model_batch)
    rev_kl = reverse_kl(logits, teacher_logits, no_model_batch)
    distil_loss = (1-lam) * for_kl + lam * rev_kl
    return distil_loss
    
def js_distance(args,logits, teacher_logits, no_model_batch, lam=0.9):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = (1-lam) * teacher_probs + lam * student_probs

    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = lam * -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

    prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss += (1-lam) * -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss
    
def tv_distance(args,logits, teacher_logits, no_model_batch):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    
    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)
    prod_probs = 0.5 * torch.masked_fill(torch.abs(teacher_probs - student_probs), inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def skewed_forward_kl(args,logits, teacher_logits, no_model_batch, lam=0.1):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = lam * teacher_probs + (1-lam) * student_probs
    mixed_logprobs = torch.log(mixed_probs)
    
    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def skewed_reverse_kl(args,logits, teacher_logits, no_model_batch, lam=0.1):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = (1-lam) * teacher_probs + lam * student_probs
    
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

