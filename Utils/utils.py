import json
import torch.nn.init
import numpy as np
import argparse, time, sys, joblib
import numpy as np
import torch.optim as optim

from Utils.data_loader import get_hyper_parameters_dataset


from Model.dkvmn_all import DKVMNMODEL
from Model.FINER import FINERMODEL
from Model.dkt_all import DKTMODEL
from Model.akt_all import AKTMODEL
from Model.sakt_all import SAKTMODEL
from Model.rkt_all import RKTMODEL
from Model.lstma_all import LSTMAMODEL
import psutil


def varible(tensor, gpu):
    if gpu >= 0:
        return torch.autograd.Variable(tensor).cuda()
    else:
        return torch.autograd.Variable(tensor)

def to_scalar(var):
    return var.view(-1).data.tolist()[0]

def save_checkpoint(state, track_list, filename):
    with open(filename + '.json', 'w') as f:
        json.dump(track_list, f)
    torch.save(state, filename + '.model')

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class LOGS(object):
    def __init__(self, params):
        self.params = params
        if (params.datascalability != 1):
            self.f = open("./Results/Log/" + params.dataset_type + "_" + str(params.datascalability) + "_" + params.model_type + "_" + str(params.seqlen) + "_" + \
                str(params.num_learningpattern) + "_" + str(params.num_followup_attempts) + "_" \
                + "_lr_" + str(params.init_lr) + "_wd_" + str(params.wd) + "_" + str(params.seed) + '_' + str(params.record_result_over_timestamp_test) + '_' + "log.txt", "w")
        else:
            self.f = open("./Results/Log/" + params.dataset_type + "_" + params.model_type + "_" + str(params.seqlen) + "_" + \
                str(params.num_learningpattern) + "_" + str(params.num_followup_attempts) + "_" \
                + "_lr_" + str(params.init_lr) + "_wd_" + str(params.wd) + "_" + str(params.seed) + '_' + str(params.record_result_over_timestamp_test) + '_' + "log.txt", "w")
        self.f.write(str(params))

    def print_with_run(self, idx, auc, accuracy, loss, run_type, cost_time=-1, test_timestamp = ""):
        if (run_type == "train"):
            print('Epoch %d/%d, train loss : %3.5f, train auc : %3.5f, train accuracy : %3.5f' % (
                idx + 1, self.params.max_iter, loss, auc, accuracy))
            self.f.write('Epoch %d/%d, train loss : %3.5f, train auc : %3.5f, train accuracy : %3.5f\n' % (
                idx + 1, self.params.max_iter, loss, auc, accuracy))
            if (cost_time != -1):
                print("Epoch %d/%d Train Cost Time: %3.5f" % (idx + 1, self.params.max_iter, cost_time))
                self.f.write("Epoch %d/%d Train Cost Time: %3.5f\n" % (idx + 1, self.params.max_iter, cost_time))
        if (run_type == "valid"):
            print('Epoch %d/%d, valid loss: %3.5f, valid auc : %3.5f, valid accuracy : %3.5f' % (
            idx + 1, self.params.max_iter, loss, auc, accuracy))
            self.f.write('Epoch %d/%d, valid loss: %3.5f, valid auc : %3.5f, valid accuracy : %3.5f\n' % (
                idx + 1, self.params.max_iter, loss, auc, accuracy))
        if (run_type == "test"):
            print("test_auc: %.4f\ttest_accuracy: %.4f\ttest_loss: %.4f\t" % (auc, accuracy, loss) + test_timestamp)
            self.f.write("test_auc: %.4f\ttest_accuracy: %.4f\ttest_loss: %.4f\t\n" % (auc, accuracy, loss) + test_timestamp)
    
    def print_best(self, best_epoch, best_valid, test_result, a=0):
        best_valid_auc, best_valid_acc, best_valid_loss = best_valid[0], best_valid[1], best_valid[2]
        test_auc, test_accuracy, test_loss = test_result[0], test_result[1], test_result[2]
        print("best outcome:-----------------------------\nbest epoch: %.4f" % (best_epoch))
        self.f.write("best outcome:-----------------------------\nbest epoch: %.4f\n" % (best_epoch))
        print(
            "valid_auc: %.4f\tvalid_accuracy: %.4f\tvalid_loss: %.4f\t" % (best_valid_auc, best_valid_acc, best_valid_loss))
        self.f.write("valid_auc: %.4f\tvalid_accuracy: %.4f\tvalid_loss: %.4f\t\n" % (
            best_valid_auc, best_valid_acc, best_valid_loss))
        print("test_auc: %.4f\ttest_accuracy: %.4f\ttest_loss: %.4f\t" % (test_auc, test_accuracy, test_loss))
        self.f.write("test_auc: %.4f\ttest_accuracy: %.4f\ttest_loss: %.4f\t\n" % (test_auc, test_accuracy, test_loss))
        if (a):
            self.f.write('result' + str(a))
    
    def print(self, info):
        print(info)
        self.f.write(info + "\n")

    def close_file(self):
        self.f.close()

def calculate_pattern_accuracies(annotated_output, dataset_type):
    dataset_dict = {
        'assist2009': [13.71, 3.33, 4.72, 2.33],
        'assist2012': [19.61, 3.74, 5.54, 2.90],
        'assist2015': [22.78, 3.44, 6.77, 3.38],
        'Algebra08': [56.54, 6.62, 6.28, 1.88],
        'HDUOJ': [2.10, 3.77, 0.61, 4.59],
        'Junyi': [33.69, 3.27, 1.27, 4.86]
    }
    pattern_correct = [0, 0, 0, 0]
    pattern_total = [0, 0, 0, 0]
    pattern_predictions = [[], [], [], []]
    pattern_real_responses = [[], [], [], []]
    for batch in annotated_output:
        for pattern_idx, pattern_data in enumerate(batch):
            predictions = np.array(pattern_data[0])
            real_responses = np.array(pattern_data[1])
            binary_predictions = (predictions >= 0.5).astype(int)
            correct = np.sum(binary_predictions == real_responses)
            pattern_correct[pattern_idx] += correct
            pattern_total[pattern_idx] += len(real_responses)
            pattern_predictions[pattern_idx].extend(predictions)
            pattern_real_responses[pattern_idx].extend(real_responses)
    accuracies = [correct / total if total > 0 else 0 for correct, total in zip(pattern_correct, pattern_total)]
    accuracies = [accuracies[i] * dataset_dict[dataset_type][i] for i in range(4)]
    accuracies[0] = accuracies[0] / (dataset_dict[dataset_type][0] + dataset_dict[dataset_type][1])
    accuracies[1] = accuracies[1] / (dataset_dict[dataset_type][0] + dataset_dict[dataset_type][1])
    accuracies[2] = accuracies[2] / (dataset_dict[dataset_type][2] + dataset_dict[dataset_type][3])
    accuracies[3] = accuracies[3] / (dataset_dict[dataset_type][2] + dataset_dict[dataset_type][3])
    return [accuracies[0] + accuracies[1], accuracies[2] + accuracies[3]]

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='the gpu will be used, e.g "0,1,2,3"')
    parser.add_argument('--seed', type=int, default=1, help='the random seed of dataset, e.g. "1/2/3/4/5"')

    parser.add_argument('--model_type', type=str, default="DKT", help='the type of model, e.g. DKVMN, or FINER')

    parser.add_argument('--max_iter', type=int, default=100, help='number of iterations')
    parser.add_argument('--maxgradnorm', type=float, default=50.0, help='maximum gradient norm')
    
    parser.add_argument('--date', type=str, default="00_00", help='Used for recording the date of result')

    # optimizor 
    parser.add_argument('--optim', type=str, default="Adagrad", help='the optimizer for training model')
    parser.add_argument('--wd', type=float, default=5e-5, help='the weight decay of optimizer')
    parser.add_argument('--init_lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0, help='learning rate decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum rate')

    # FINER model
    parser.add_argument('--generate_followup_trends', type=int, default=1,
                        help='Generate followup_trends or not e.g. True or False') 
    parser.add_argument('--num_learningpattern', type=int, default=2,
                        help='learning_trend_embed_dim')
    parser.add_argument('--num_followup_attempts', type=int, default=2,
                        help='learning_trend_embed_dim')
    # 30,50,50
    parser.add_argument('--batch_size', type=int, default=128, help='the batch size')
    parser.add_argument('--q_embed_dim', type=int, default=50, help='question embedding dimensions')
    parser.add_argument('--qa_embed_dim', type=int, default=200, help='answer and question embedding dimensions')
    parser.add_argument('--final_fc_dim', type=int, default=100, help='hidden state dim for final fc layer')
    parser.add_argument('--memory_size', type=int, default=20, help='memory size for lstm')

    # further analysis 
    parser.add_argument('--vis_pred', type=int, default=0, help='visualization_pred') 
    parser.add_argument('--record_result_over_timestamp_test', type=int, default=0, help='record_result_over_timestamp_test')
    parser.add_argument('--test_differentpattern', type=int, default=0, help='test_differentpattern')
    parser = get_hyper_parameters_dataset(parser)

    params = parser.parse_args()
    return params


def model_select(params):
    # param_dict
    params.lr = params.init_lr

    if ("FINER" in params.model_type):
        params.memory_key_state_dim = params.q_embed_dim
        params.memory_value_state_dim = params.qa_embed_dim
        model = FINERMODEL(params)

    if (params.model_type == "DKVMN"):
        params.memory_key_state_dim = params.q_embed_dim
        params.memory_value_state_dim = params.qa_embed_dim
        model = DKVMNMODEL(params)

    if (params.model_type == "DKT"):
        model = DKTMODEL(params)

    if (params.model_type == "AKT"):
        model = AKTMODEL(n_question=params.n_question,
                         n_pid=-1,
                         d_model=200,
                         n_blocks=1,
                         kq_same=1,
                         dropout=0.05,
                         model_type='akt',
                         final_fc_dim=params.final_fc_dim,
                         n_heads=8,
                         d_ff=2048,
                         l2=1e-5,
                         separate_qa=False)

    if (params.model_type == "SAKT"):
        model = SAKTMODEL(params)

    if (params.model_type == "RKT"):
        model = RKTMODEL(num_items=params.n_question,
                         embed_size=200,
                         num_attn_layers=1,
                         num_heads=5,
                         encode_pos=False,
                         max_pos=10,
                         drop_prob=0.2)
        
    if (params.model_type == 'LSTMA'):
        model = LSTMAMODEL(params)
    
    total_memory_usage = 0
    for param in model.parameters():
        total_memory_usage += param.element_size() * param.nelement()
    print("Total memory usage of " + params.model_type + f": {total_memory_usage/(1024 * 1024):.2f} MB")
    return model

def optim_select(params, model):
    if (params.optim == "SGD"):
        optimizer = optim.SGD(params=model.parameters(), lr=params.lr, momentum=params.momentum, weight_decay = params.wd)
    
    if (params.optim == "Adam"):
        optimizer = optim.Adam(params=model.parameters(), lr=params.lr, betas=(0.9, 0.99), weight_decay=params.wd)

    if (params.optim == "Adagrad"):
        optimizer = optim.Adagrad(params=model.parameters(), lr=params.lr, weight_decay=params.wd, lr_decay=params.lr_decay)

    return optimizer
