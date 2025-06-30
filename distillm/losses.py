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
    
    # mask为True的区域对应教师的低置信度（尾部）区域
    # mask为False的区域对应教师的高置信度（头部）区域
    mask = cum_sum > mu
    mask[:,:,0]=False #第一个概率一定要置False，对应第一个概率>0.5时mask全True
    #   ghead,教师低置信度（尾部）区域的误差总和
    s1 = torch.masked_fill(errors, mask, 0.0).sum(dim=-1)
    #   gtail,教师高置信度（头部）区域的误差总和
    s2 = torch.masked_fill(errors, ~mask, 0.0).sum(dim=-1)
    total_errors = s1 + s2
    total_errors = torch.where(total_errors < 1e-6, torch.full_like(total_errors, 1e-6), total_errors)

    return s1/total_errors, s2/total_errors

# def get_kl(args,logits, teacher_logits, inf_mask, mask, ratio=None,mu=0.5):
#     tea_temp = args.tea_temp
#     stu_temp = args.stu_temp
#     #ratio: [B,L]
#     # p
#     teacher_probs = F.softmax(teacher_logits,dim=-1, dtype=torch.float32)
#     # log p
#     teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
#     # p*logp
#     teacher_prod_probs =  torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
#     # Σp*logp
#     teacher_x =  torch.sum(teacher_prod_probs, dim=-1).view(-1)
#     #logq
#     logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
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
    teacher_probs = F.softmax(teacher_logits/tea_temp, dim=-1, dtype=torch.float32)
    # log p
    teacher_logprobs = F.log_softmax(teacher_logits/tea_temp, dim=-1, dtype=torch.float32)
    # p*logp
    teacher_prod_probs =  torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
    # Σp*logp
    teacher_x =  torch.sum(teacher_prod_probs, dim=-1).view(-1)
    #logq
    logprobs = F.log_softmax(logits/stu_temp, dim=-1, dtype=torch.float32)
    #p*logq
    prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
    #Σp*logq
    x = torch.sum(prod_probs, dim=-1).view(-1) # [B,L]->[BL]

    if ratio == None:
        distil_loss = torch.sum((teacher_x-x) * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    else:
        distil_loss = torch.sum((teacher_x-x) * ratio.view(-1) * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def TDKL_HEG_FKL(args, logits, teacher_logits, no_model_batch, head_error_threshold=0.3):
    """
    TDKL损失函数，结合了头部误差引导的反向KL散度强化 (HEG-RKL) 机制。
    当学生模型在教师高置信度（头部）区域的误差比例超过阈值时，
    对反向KL散度（RKL）施加额外的温度缩放。

    Args:
        args: 包含 tea_temp 和 stu_temp 的参数对象。
        logits (torch.Tensor): 学生模型的原始logits，形状为 [batch, seq, vocab]。
        teacher_logits (torch.Tensor): 教师模型的原始logits，形状为 [batch, seq, vocab]。
        no_model_batch (dict): 包含"label"键的字典，用于标记有效token（-100为无效）。
        head_error_threshold (float): 触发RKL缩放的l_ratio阈值。
                                      例如，0.3表示当30%或更多的误差集中在教师头部区域时，
                                      激活RKL的额外缩放。
    Returns:
        torch.Tensor: 计算出的TDKL损失值。
    """
    inf_mask = torch.isinf(logits) # [batch, seq, vocab]
    mask = (no_model_batch["label"]!= -100).int() # [batch, seq] (有效token掩码)

    # 计算头部和尾部误差比例
    # h_ratio: 误差在教师低置信度（尾部）区域的比例
    # l_ratio: 误差在教师高置信度（头部）区域的比例
    h_ratio, l_ratio = get_ratio(teacher_logits, logits) #

    # 计算 FKL (KL(P_teacher || P_student))
    # FKL 旨在让学生模仿教师的整体分布，尤其强调教师高概率区域
    # 使用 h_ratio (尾部误差比例) 加权 FKL
    rkl_per_token = get_kl(teacher_logits, logits, inf_mask, mask, h_ratio) #

    # 计算 RKL (KL(P_student || P_teacher))
    # RKL 旨在防止学生对教师低概率区域过度自信，强调教师低概率区域
    # 使用 l_ratio (头部误差比例) 加权 RKL
    fkl_per_token = get_kl(logits, teacher_logits, inf_mask, mask, l_ratio) #

    # 初始化RKL的缩放因子，默认为1.0（不缩放）
    rkl_scaling_factor = torch.ones_like(rkl_per_token) #

    # 定义条件：何时对RKL应用额外的温度缩放
    # 如果学生在教师高置信度（头部）区域的误差比例 (l_ratio) 超过阈值，则满足条件
    condition_met = (l_ratio > head_error_threshold).view(-1) # 布尔张量
    # 在满足条件的token位置应用组合温度缩放因子 (stu_temp * tea_temp)
    rkl_scaling_factor[condition_met] = args.stu_temp^2

    # 将缩放因子应用于FKL
    fkl_loss_scaled = fkl_per_token * rkl_scaling_factor

    # 组合FKL和缩放后的RKL
    # 确保只对有效token进行求和和平均
    valid_tokens_mask = mask.view(-1).bool() # 布尔张量
    
    # 仅对有效token计算总损失
    total_rkl_loss = torch.sum(rkl_per_token[valid_tokens_mask])
    total_fkl_loss_scaled = torch.sum(fkl_loss_scaled[valid_tokens_mask])
    
    num_valid_tokens = torch.sum(mask).item()

    if num_valid_tokens > 0:
        distil_loss = (total_rkl_loss + total_fkl_loss_scaled) / num_valid_tokens
    else:
        distil_loss = torch.tensor(0.0, device=logits.device) # 没有有效token时损失为0

    return distil_loss

def TDKL(args, logits,teacher_logits,no_model_batch):
    inf_mask = torch.isinf(logits)
    mask = (no_model_batch["label"]!=-100).int()
    h_ratio, l_ratio = get_ratio(teacher_logits, logits)
    distil_loss =get_kl(args,teacher_logits, logits, inf_mask, mask, h_ratio) + (args.stu_temp**2)*get_kl(args,logits,teacher_logits, inf_mask, mask, l_ratio)
    return distil_loss
def AKL(args, logits,teacher_logits,no_model_batch):
    inf_mask = torch.isinf(logits) # [batch, seq, vocab]
    mask = (no_model_batch["label"] != -100).int() # [batch, seq]
    h_ratio, l_ratio = get_ratio(teacher_logits, logits)
    distil_loss =get_kl(args,teacher_logits, logits, inf_mask, mask, h_ratio) + (args.stu_temp**2)*get_kl(args,logits,teacher_logits, inf_mask, mask, l_ratio)
    # distil_loss = get_kl(args,teacher_logits, logits, inf_mask, mask, h_ratio) + (args.stu_temp*args.tea_temp)*get_kl(args,logits,teacher_logits, inf_mask, mask, l_ratio)
    # distil_loss = (args.stu_temp*args.tea_temp)*(get_kl(args,teacher_logits, logits, inf_mask, mask, h_ratio) + get_kl(args,logits,teacher_logits, inf_mask, mask, l_ratio))
    # distil_loss = get_kl(args,teacher_logits, logits, inf_mask, mask, h_ratio) + get_kl(args,logits,teacher_logits, inf_mask, mask, l_ratio)
    return distil_loss

def compute_dtkd_loss(logits_student, logits_teacher, temperature, alpha, beta, warmup, epoch):
    """
    计算 DTKD 损失
    
    参数:
        logits_student: 学生模型输出 [B, C]
        logits_teacher: 教师模型输出 [B, C]
        temperature: 温度参数
        alpha: ourskd loss 权重
        beta: kd loss 权重
        warmup: 预热周期数
        epoch: 当前训练轮次

    返回:
        loss_dtkd: 最终的 DTKD 损失
        losses_dict: 包含各个子项损失的字典
    """
    # DTKD Loss
    reference_temp = temperature
    logits_student_max, _ = logits_student.max(dim=1, keepdim=True)
    logits_teacher_max, _ = logits_teacher.max(dim=1, keepdim=True)

    logits_student_temp = 2 * logits_student_max / (logits_teacher_max + logits_student_max) * reference_temp
    logits_teacher_temp = 2 * logits_teacher_max / (logits_teacher_max + logits_student_max) * reference_temp

    ourskd = torch.nn.KLDivLoss(reduction='none')(
        torch.nn.functional.log_softmax(logits_student / logits_student_temp, dim=1),
        torch.nn.functional.softmax(logits_teacher / logits_teacher_temp, dim=1)
    )
    loss_ourskd = (ourskd.sum(1, keepdim=True) * logits_teacher_temp * logits_student_temp).mean()

    # Vanilla KD Loss
    vanilla_temp = temperature
    kd = torch.nn.KLDivLoss(reduction='none')(
        torch.nn.functional.log_softmax(logits_student / vanilla_temp, dim=1),
        torch.nn.functional.softmax(logits_teacher / vanilla_temp, dim=1)
    )
    loss_kd = (kd.sum(1, keepdim=True) * vanilla_temp ** 2).mean()

    # CrossEntropy Loss
    loss_ce = torch.nn.CrossEntropyLoss()(logits_student, logits_teacher.argmax(dim=1))  # 假设使用教师标签

    # 总损失
    warmup_factor = min(epoch / warmup, 1.0)
    loss_dtkd = warmup_factor * (alpha * loss_ourskd + beta * loss_kd) + loss_ce

    losses_dict = {
        "loss_dtkd": loss_dtkd,
        "loss_ourskd": loss_ourskd,
        "loss_kd": loss_kd,
        "loss_ce": loss_ce
    }

    return loss_dtkd, losses_dict


def forward_kl(args,logits, teacher_logits, no_model_batch):
    # tea_temp = args.tea_temp 
    # stu_temp = args.stu_temp
    # # p
    # teacher_probs = F.softmax(teacher_logits/tea_temp, dim=-1, dtype=torch.float32)
    # inf_mask = torch.isinf(logits)
    # #log q
    # student_logprobs = F.log_softmax(logits/stu_temp, dim=-1, dtype=torch.float32)
    # prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
    # x = torch.sum(prod_probs, dim=-1).view(-1)
    # mask = (no_model_batch["label"] != -100).int()
    # distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    # return distil_loss
    inf_mask = torch.isinf(logits) # [batch, seq, vocab]
    mask = (no_model_batch["label"] != -100).int() # [batch, seq]
    distill_loss = (args.stu_temp**2)*get_kl(args,logits,teacher_logits, inf_mask, mask)
    return distill_loss
def reverse_kl(args,logits, teacher_logits, no_model_batch):
    # student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    # student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    # teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    # inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)
    # prod_probs = torch.masked_fill(student_probs * teacher_logprobs, inf_mask, 0)
    # prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    # x = torch.sum(prod_probs, dim=-1).view(-1)
    # mask = (no_model_batch["label"] != -100).int()
    # distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    # return distil_loss
    inf_mask = torch.isinf(logits) # [batch, seq, vocab]
    mask = (no_model_batch["label"] != -100).int() # [batch, seq]
    distill_loss = (args.stu_temp**2)*get_kl(args,teacher_logits,logits,inf_mask, mask)
    return distill_loss
def symmetric_kl(args,logits, teacher_logits, no_model_batch, lam=0.9):
    for_kl = forward_kl(logits, teacher_logits, no_model_batch)
    rev_kl = reverse_kl(logits, teacher_logits, no_model_batch)
    distil_loss = (1-lam) * for_kl + lam * rev_kl
    return distil_loss
    
def js_distance(args,logits, teacher_logits, no_model_batch, lam=0.9):
    tea_temp = args.tea_temp
    stu_temp = args.stu_temp
    teacher_probs = F.softmax(teacher_logits/tea_temp, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits/stu_temp, dim=-1, dtype=torch.float32)
    mixed_probs = (1-lam) * teacher_probs + lam * student_probs

    teacher_logprobs = F.log_softmax(teacher_logits/tea_temp, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits/stu_temp, dim=-1, dtype=torch.float32)
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
    tea_temp = args.tea_temp
    stu_temp = args.stu_temp
    teacher_probs = F.softmax(teacher_logits/tea_temp, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits/stu_temp, dim=-1, dtype=torch.float32)
    mixed_probs = lam * teacher_probs + (1-lam) * student_probs
    mixed_logprobs = torch.log(mixed_probs)
    
    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def skewed_reverse_kl(args,logits, teacher_logits, no_model_batch, lam=0.1):
    tea_temp = args.tea_temp
    stu_temp = args.stu_temp
    teacher_probs = F.softmax(teacher_logits/tea_temp, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits/stu_temp, dim=-1, dtype=torch.float32)
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

