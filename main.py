import torch, time
from Utils.run import run_one_epoch
from Utils.utils import get_parser, model_select, optim_select
from Utils.utils import LOGS, calculate_pattern_accuracies
from Utils.data_loader import load_DataSet
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def main():
    #region Step 0: Load Parameters and Data
    params = get_parser()
    logs = LOGS(params)
    DataSet, FollowupTrends = load_DataSet(params=params, logs = logs)
    FPT_train = FPT_valid = FPT_test = []
    if "FINER" in params.model_type:
        FPT_train = FollowupTrends['train']
        FPT_valid = FollowupTrends['valid']
        FPT_test  = FollowupTrends['test']
    #endregion

    #region Step 1: Model Selection
    model = model_select(params=params)
    model.init_embeddings()
    # model.init_params()
    if params.gpu >= 0:
        print('device: ' + str(params.gpu))
        torch.cuda.set_device(params.gpu)
        model.cuda()
    #endregion

    #region Step 2: Optimizer Selection
    optimizer = optim_select(params, model)
    #endregion
    
    #region Step 3: Training and Validation
    all_train_loss, all_train_accuracy, all_train_auc = {}, {}, {}
    all_valid_loss, all_valid_accuracy, all_valid_auc = {}, {}, {}
    best_valid_auc = 0

    for idx in range(params.max_iter):
        #region Step 3.1: Training
        start_time = time.time()
        train_loss, train_accuracy, train_auc, _  = run_one_epoch("train", model, params, DataSet['train'], FPT_train, optimizer = optimizer)
        logs.print_with_run(idx, train_auc, train_accuracy, train_loss, "train", cost_time = time.time() - start_time)
        #endregion

        #region Step 3.2: Validation
        valid_loss, valid_accuracy, valid_auc, _ = run_one_epoch("valid", model, params, DataSet['valid'], FPT_valid)
        logs.print_with_run(idx, valid_auc, valid_accuracy, valid_loss, "valid")

        all_train_auc[idx + 1] , all_train_accuracy[idx + 1], all_train_loss[idx + 1] = train_auc,  train_accuracy, train_loss
        all_valid_loss[idx + 1], all_valid_accuracy[idx + 1], all_valid_auc[idx + 1]  = valid_loss, valid_accuracy, valid_auc
        #endregion

        #region Step 3.3: Testing
        # output the epoch with the best validation auc
        if valid_auc > best_valid_auc:
            print(best_valid_auc, valid_auc)
            best_valid_auc, best_valid_acc, best_valid_loss, best_epoch,  = valid_auc, valid_accuracy, valid_loss, idx + 1 
            test_loss, test_accuracy, test_auc, add_info = run_one_epoch("test", model, params, DataSet['test'], FPT_test)
            #region further analysis
            if (params.test_differentpattern and 'annotated_output' in add_info):
                pattern_accuracies = calculate_pattern_accuracies(add_info['annotated_output'], params.dataset_type)
                print(pattern_accuracies)
                if not os.path.exists("./Results/Pattern/"):
                    os.makedirs("./Results/Pattern/")
                with open("./Results/Pattern/pattern_results.txt", "a") as f:
                    f.write(f"{params.dataset_type},{params.model_type},{','.join(map(str, pattern_accuracies))}\n")

            if (params.vis_pred):
                train_loss, train_accuracy, train_auc, add_info  = run_one_epoch("test", model, params, DataSet['train'], FPT_train)
                if not os.path.exists("./Results/Weight/"):
                    os.makedirs("./Results/Weight/")    
                if ("FINER" in params.model_type):
                    np.save("./Results/Weight/" + params.model_type + "_pred_all.npy", np.array(add_info['pred']))
                    np.save("./Results/Weight/" + params.model_type + "_weight_trends.npy", np.array(add_info['weight_of_FPTs']))
                elif ("DKT" == params.model_type):
                    train_loss, train_accuracy, train_auc, add_info  = run_one_epoch("test", model, params, DataSet['train'], FPT_train)
                    np.save("./Results/Weight/" + params.model_type + "_pred.npy", np.array(add_info['DKT_pred']))

            test_auc_over_timestamp = []
            if params.record_result_over_timestamp_test == 1 :
                for i in range(params.seqlen):
                    test_loss, test_accuracy, test_auc, _ = run_one_epoch("test", model, params, DataSet['test'], FPT_test, test_timestamp=i)
                    logs.print_with_run(idx, test_auc, test_accuracy, test_loss, "test", i)
                    test_auc_over_timestamp.append(test_auc)
            #endregion
            logs.print_best(best_epoch, [best_valid_auc, best_valid_acc, best_valid_loss], [test_auc, test_accuracy, test_loss], test_auc_over_timestamp)
        #endregion
    #endregion
    logs.print_best(best_epoch, [best_valid_auc, best_valid_acc, best_valid_loss], [test_auc, test_accuracy, test_loss])
    logs.close_file()

if __name__ == "__main__":
    main()
