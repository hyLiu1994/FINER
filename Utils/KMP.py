import time
import numpy as np

def search_FPTs_in_ITS(Dataset_user_Sequence, query_sequence, num_followup_attempts, question_num, target_question):
    frequency = [[0,0] for idx in range(num_followup_attempts)]

    for idx_user in range(len(Dataset_user_Sequence)):
        for idx_seq in range(len(Dataset_user_Sequence[idx_user])):
            match = True
            for idx_q_seq in range(len(query_sequence)):
                if (idx_seq + idx_q_seq >= len(Dataset_user_Sequence[idx_user])):
                    match = False
                    break
                if (Dataset_user_Sequence[idx_user][idx_seq + idx_q_seq] != query_sequence[idx_q_seq]):
                    match = False
            if (match): 
                for idx_follwup in range(1, num_followup_attempts + 1):
                    if (idx_seq + len(query_sequence) + idx_follwup >= len(Dataset_user_Sequence[idx_user])):
                        break
                    if (Dataset_user_Sequence[idx_user][idx_seq + len(query_sequence) + idx_follwup] == target_question or 
                        Dataset_user_Sequence[idx_user][idx_seq + len(query_sequence) + idx_follwup] == target_question + question_num):
                        frequency[idx_follwup-1][Dataset_user_Sequence[idx_user][idx_seq + len(query_sequence) + idx_follwup]>=question_num] += 1
                
    rho = [[frequency[idx][0]/(frequency[idx][0] + frequency[idx][1] + 1e-9), 
            frequency[idx][1]/(frequency[idx][0] + frequency[idx][1] + 1e-9)] for idx in range(len(frequency)) ]
    return frequency, rho

def load_followup_trend_KMP(Dataset_user_Sequence, Target_user_Sequence, num_learningpattern, num_followup_attempts, question_num):
    weight_list  = []  # [target_dataset_num, sequence_len, num_learningpattern, num_followup_attempts_weight]
    p_list       = []  # [target_dataset_num, sequence_len, num_learningpattern, num_followup_attempts_probability]
    dep_list     = []  # [target_dataset_num, sequence_len, num_learningpattern, num_followup_attempts_dep]

    # Target_user_Sequence: [data_num, seqlen]
    for i in range(len(Target_user_Sequence)):
        output_weight = []
        output_p = []
        output_dep = []
        time_start_1 = time.time()
        time_1_list = []
        for idx_seq in range(-1, len(Target_user_Sequence[i])-1):
            output_weight_tmp = []
            output_p_tmp = []
            output_dep_tmp = []

            time_start_2 = time.time()
            for idx_learningpatttern in range(num_learningpattern):
                # print(i, idx_seq, Target_user_Sequence[i][idx_seq-idx_learningpatttern:idx_seq])
                target_question = Target_user_Sequence[i][idx_seq + 1]
                if (target_question >= question_num):
                    target_question -= question_num
                if (idx_seq == -1):
                    query_sequence = []
                else:
                    query_sequence = Target_user_Sequence[i][idx_seq-idx_learningpatttern:idx_seq]
                frequency, rho = search_FPTs_in_ITS(Dataset_user_Sequence, query_sequence, num_followup_attempts, 
                                                    question_num, target_question)
                output_weight_tmp.append(frequency)
                output_p_tmp.append(rho)
                output_dep_tmp.append(len(query_sequence))
            output_weight.append(output_weight_tmp)
            output_p.append(output_p_tmp)
            output_dep.append(output_dep_tmp)
            time_1_list.append(time.time()-time_start_2)
            print("The time cost of one sequence", time.time()-time_start_2, np.mean(time_1_list), len(Target_user_Sequence), len(Target_user_Sequence[i]))
        print("The time cost of one sequence", time.time()-time_start_1, len(Target_user_Sequence))

        weight_list.append(output_weight)
        p_list.append(output_p)
        dep_list.append(output_dep)

    return np.array(weight_list).astype(np.int32), np.array(p_list), np.array(dep_list).astype(np.int32), []
