import numpy as np
import math
import torch
from torch import nn
import Model.utils as utils
from sklearn import metrics

def annotate_tensor(original_tensor, question_num):
    batch_size, seq_len = original_tensor.shape
    annotated_tensor = torch.zeros_like(original_tensor)

    prev1 = torch.roll(original_tensor, shifts=1, dims=1)
    prev2 = torch.roll(original_tensor, shifts=2, dims=1)
    next1 = torch.roll(original_tensor, shifts=-1, dims=1)

    mask = torch.ones_like(original_tensor, dtype=torch.bool)
    mask[:, :2] = False
    mask[:, -1] = False
    # 1111 pattern
    condition_1111 = (original_tensor == prev1) & (original_tensor == prev2) & (original_tensor == next1) & (original_tensor > question_num)
    annotated_tensor[condition_1111 & mask] = 1
    # 1110 pattern
    condition_1110 = (original_tensor == prev1) & (original_tensor == prev2) & (original_tensor == next1 + question_num)
    annotated_tensor[condition_1110 & mask] = 2
    # 1101 pattern
    condition_1101 = (original_tensor == prev1 - question_num) & (original_tensor == prev2 - question_num) & (original_tensor == next1 - question_num)
    annotated_tensor[condition_1101 & mask] = 3
    # 1100 pattern
    condition_1100 = (original_tensor == prev1 - question_num) & (original_tensor == prev2 - question_num) & (original_tensor == next1)
    annotated_tensor[condition_1100 & mask] = 4
    return annotated_tensor

def get_one_batch_data(idx, params, q_data, qa_data, FPTs):

    # sufs_batch_seq=similar_user_future_sequence[idx * params.batch_size:(idx + 1) * params.batch_size]
    input_q = utils.varible(torch.LongTensor(
        q_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]), params.gpu)
    input_qa = utils.varible(torch.LongTensor(
        qa_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]), params.gpu)
    # sufs_batch_seq=similar_user_future_sequence[idx * params.batch_size:(idx + 1) * params.batch_size]

    target = qa_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]
    target = (target - 1) / params.n_question
    target = np.floor(target)

    input_FPTs = []
    if (len(FPTs)):
        input_similar_user_future_weight_sequence = utils.varible(torch.LongTensor(
            FPTs[0][idx * params.batch_size:(idx + 1) * params.batch_size]), params.gpu)
        input_similar_user_future_p_sequence      = utils.varible(torch.FloatTensor(
            FPTs[1][idx * params.batch_size:(idx + 1) * params.batch_size]), params.gpu)
        input_similar_user_future_dep_sequence    = utils.varible(torch.LongTensor(
            FPTs[2][idx * params.batch_size:(idx + 1) * params.batch_size]), params.gpu)
        input_similar_user_future_idx_sequence    = utils.varible(torch.LongTensor(
            FPTs[3][idx * params.batch_size:(idx + 1) * params.batch_size]), params.gpu)
        input_FPTs = [input_similar_user_future_weight_sequence, 
                      input_similar_user_future_p_sequence, 
                      input_similar_user_future_dep_sequence, 
                      input_similar_user_future_idx_sequence]

    return input_q, input_qa, target, input_FPTs

def run_one_epoch(run_type, model, params, data, FPTs, optimizer=None, test_timestamp=-1):
    q_data, qa_data  = data[0], data[1]
    N = int(math.floor(len(q_data) / params.batch_size))
    pred_list, target_list, epoch_loss = [], [], 0
    add_info = {
        'weight_of_FPTs': [],
        'pred':[],
        'DKT_pred':[],
        'annotated_output':[]
    }

    if (run_type == "train"):
        model.train()
    else:
        model.eval()

    for idx in range(N):
        input_q, input_qa, target, input_FPTs = get_one_batch_data(idx, params, q_data, qa_data, FPTs)
        if (params.test_differentpattern):
            annotated_xxxx = annotate_tensor(input_qa, params.n_question)

        if params.model_type == 'SAKT':
            target = utils.varible(torch.FloatTensor(target), params.gpu)
        else:
            target = utils.varible(torch.FloatTensor(target), params.gpu)
            target_to_1d = torch.chunk(target, params.batch_size, 0)
            target_1d = torch.cat([target_to_1d[i] for i in range(params.batch_size)], 1)
            target_1d = target_1d.permute(1, 0)

        if (run_type == "train"):
            model.zero_grad()

        if (not len(FPTs)):
            if params.model_type in ["DKT", "DKVMN", "AKT", "LSTMA"]:
                if (params.test_differentpattern):
                    loss, filtered_pred, filtered_target, add_info_one, annotated_output = model.forward(input_q, input_qa, target_1d, test_timestamp, annotated_qa=annotated_xxxx)
                else:
                    loss, filtered_pred, filtered_target, add_info_one = model.forward(input_q, input_qa, target_1d, test_timestamp)
                if (params.vis_pred):
                    add_info['DKT_pred'].append(add_info_one.cpu().detach().numpy())
            elif params.model_type == 'SAKT':
                loss, filtered_pred, filtered_target = model.forward(input_q, input_qa, target, test_timestamp)
            elif params.model_type == 'RKT':
                if (params.test_differentpattern):
                    seq_len = input_q.shape[1]
                    rel = np.zeros((seq_len,seq_len))
                    rel = torch.Tensor(rel).cuda()
                    timestamp = torch.arange(seq_len).cuda()
                    loss, filtered_pred, filtered_target, add_info_one, annotated_output = model.forward(input_q, ((input_qa - input_q) // params.n_question), input_q, rel, timestamp, target, test_timestamp, annotated_qa=annotated_xxxx)
                else:
                    seq_len = input_q.shape[1]
                    rel = np.zeros((seq_len,seq_len))
                    rel = torch.Tensor(rel).cuda()
                    timestamp = torch.arange(seq_len).cuda()
                    loss, filtered_pred, filtered_target = model.forward(input_q, ((input_qa - input_q) // params.n_question), input_q, rel, timestamp, target, test_timestamp)
            else:
                loss, filtered_pred, filtered_target = model.forward(input_q, input_qa, target_1d, test_timestamp)
        else:
            if (params.test_differentpattern):
                loss, filtered_pred, filtered_target, add_info_one, annotated_output = model.forward(input_q, input_qa, input_FPTs, 
                                                                 target_1d, test_timestamp, annotated_qa=annotated_xxxx)
            else:
                loss, filtered_pred, filtered_target, add_info_one = model.forward(input_q, input_qa, input_FPTs, 
                                                                 target_1d, test_timestamp)
            if (params.vis_pred):
                add_info['weight_of_FPTs'].append(add_info_one[0].cpu().detach().numpy())
                add_info['pred'].append(add_info_one[1].cpu().detach().numpy())

        if (run_type == "train"):
            loss.backward(torch.ones_like(loss))
            # loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), params.maxgradnorm)
            optimizer.step()

        right_target = np.asarray(filtered_target.data.tolist())
        right_pred = np.asarray(filtered_pred.data.tolist())

        if (params.test_differentpattern and 'annotated_output' in add_info):
            add_info['annotated_output'].append(annotated_output)

        pred_list.append(right_pred)
        target_list.append(right_target)

        if (params.vis_pred):
            if (len(FPTs)):
                add_info['weight_of_FPTs'].append(add_info_one[0].cpu().detach().numpy())
                add_info['pred'].append(add_info_one[1].cpu().detach().numpy())
            elif (params.model_type == "DKT"):
                add_info['DKT_pred'].append(add_info_one.cpu().detach().numpy())

        epoch_loss += utils.to_scalar(loss)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)
    auc = metrics.roc_auc_score(all_target, all_pred)

    if (params.vis_pred):
        if (len(FPTs)):
            add_info['weight_of_FPTs'] = np.concatenate(add_info['weight_of_FPTs'], axis = 0)
            add_info['pred'] = np.concatenate(add_info['pred'], axis = 0)
        elif (params.model_type == "DKT"):
            add_info['DKT_pred'] = np.concatenate(add_info['DKT_pred'], axis = 0)


    all_pred_01 = [0 for i in range(len(all_pred))]
    all_pred_01 = np.where(all_pred >= 0.5, 1.0, all_pred_01)
    # all_pred_01[all_pred >= 0.5] = 1.0
    # all_pred_01[all_pred < 0.5] = 0.0
    accuracy = metrics.accuracy_score(all_target, all_pred_01)

    return epoch_loss/N, accuracy, auc, add_info