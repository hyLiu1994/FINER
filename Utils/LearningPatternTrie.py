import copy, time
from queue import Queue
from typing import List, Dict, Iterable
import numpy as np
from tqdm import tqdm

# from data_loader import DATA

class LearningPatternTrie(object):

    class Node(object):

        def __init__(self, name: str):
            self.idx = 0
            self.dep = 0
            self.name = name  
            self.children = {}  
            self.weight = {}
            self.fail = None  
            self.Myweight = 0

            #self.qa_list=[]
            self.sample_weight_list = {}
            self.sample_probability_list = {}

    def __init__(self, keywords: Iterable[list] = None):
        self.node_num = 0
        self.root = self.Node("-1")
        self.root.dep = 0
        self.root.idx = self.node_num
        #self.root.qa_list = []
        self.node_num += 1
        self.finalized = False
        # keywords: [data_num, seqlen]
        if keywords is not None:
            for keyword in keywords:
                # keyword: [seqlen]
                self.add(keyword)

    def add(self, keyword: list, num_followup_attempts, num_learningpattern):
        if self.finalized:
            raise RuntimeError('The tree has been finalized!')
        node = self.root
        pre_num = ['None', 'None']
        for num in keyword:
            if (num == pre_num[1] and pre_num[0] == pre_num[1]):
                continue
            if num == '0.0' or num == '0':
                break
            else:
                if num not in node.children:
                    node.children[num] = self.Node(num)
                    # node.children[num].idx = self.node_num
                    node.children[num].dep = node.dep + 1
                    #node.children[num].qa_list = node.qa_list + [num]
                    # self.node_num += 1
                    node.weight[num] = 1
                else:
                    node.weight[num] += 1
                    
                node = node.children[num]
            if (node.dep > num_followup_attempts + num_learningpattern + 2):
                break
            pre_num[0] = pre_num[1]
            pre_num[1] = num

    def finalize(self):
        queue = Queue()
        queue.put(self.root)
        self.node_num = 1
        dep_node_num = {}
        while not queue.empty():
            node = queue.get()
            node.idx = self.node_num
            if (node.dep not in dep_node_num):
                dep_node_num[node.dep] = 0
            dep_node_num[node.dep] += 1
            self.node_num += 1
            for char in node.children:
                child = node.children[char]
                f_node = node.fail
                while f_node is not None:
                    if char in f_node.children:
                        f_child = f_node.children[char]
                        child.fail = f_child
                        # if f_child.exist:
                        #     child.exist.extend(f_child.exist)
                        break
                    f_node = f_node.fail
                if f_node is None:
                    child.fail = self.root
                queue.put(child)
        self.finalized = True
        # print(dep_node_num)

    def get_sample_info(self, node, max_dep):
        if (len(node.children.keys())==0):
            node.Myweight = 0
            return {}

        node.sample_weight_list = {1:{}}
        for child_id in node.weight.keys():
            child_dict = self.get_sample_info(node.children[child_id], max_dep)
            node.sample_weight_list[1][child_id] = node.weight[child_id]
            for dep_idx in child_dict.keys():
                if (dep_idx > max_dep - 1):
                    continue
                if (dep_idx + 1 not in node.sample_weight_list):
                    node.sample_weight_list[dep_idx + 1] = {}
                for node_name in child_dict[dep_idx].keys():
                    if (node_name not in node.sample_weight_list[dep_idx + 1]):
                        node.sample_weight_list[dep_idx + 1][node_name] = 0
                    node.sample_weight_list[dep_idx + 1][node_name] += child_dict[dep_idx][node_name]
        
        node.sample_probability_list = {}
        node.Myweight = sum(node.sample_weight_list[1].values())
        for dep_idx in node.sample_weight_list.keys():
            node.sample_probability_list[dep_idx] = np.array(list(node.sample_weight_list[dep_idx].values())) / sum(node.sample_weight_list[dep_idx].values())
        
        return node.sample_weight_list

    def search_in_4(self, Target_user_Sequence: list, similar_user: int, future_sequence: int,question_num: int) -> List[int]:
        
        node = self.root
        # similar [sequence_len, similar_user_num, future_sequence_len, 2]
        similar = []
        weight_all = []
        p_all = []
        dep_all = []
        # node_idx_all [sequence_len, similar_user_num]
        node_idx_all = []

        # [similar_user_num, future_sequence_len, 2]
        result = []
        weight = []
        P = []
        dep_list = []
        # [similar_user_num, future_sequence_len, 2]
        idx_all = []

        # [similar_user_num, future_sequence_len, 2]
        zeros_padding = np.array([[[0.0, 0.0] for j in range(future_sequence)] for i in range(similar_user)])
        # [similar_user_num]
        zeros_padding_idx_all = np.array([0 for i in range(similar_user)])

        pre_char = ["None", "None"]
        for i in range(len(Target_user_Sequence)):
            char = str(Target_user_Sequence[i])

            if (char == '0' or char == "0.0"):
                similar.append(zeros_padding)
                weight_all.append(zeros_padding)
                p_all.append(zeros_padding)
                dep_all.append(zeros_padding)
                node_idx_all.append(zeros_padding_idx_all)
                continue
            
            # print(char, pre_char, i, len(Target_user_Sequence))
            if str(Target_user_Sequence[i]) == '0.0' or str(Target_user_Sequence[i]) == '0':
                now_next_qa = -1
            else:
                now_next_qa = Target_user_Sequence[i]

            if (char == pre_char[1] and pre_char[1] == pre_char[0]):
                similar.append(result)
                weight_all.append(weight)
                p_all.append(P)
                dep_all.append(dep_list)
                node_idx_all.append(idx_all)
                continue

            node_tmp = node

            # [similar_user_num, future_sequence_len, 2]
            result = []
            weight = []
            P = []
            dep_list = []
            # [similar_user_num]
            idx_all = []
            
            if now_next_qa == -1:
                now_next_qtf = [0.0, 0.0]
            else:
                if (now_next_qa > question_num):
                    now_next_qtf = [now_next_qa - question_num, now_next_qa]
                else:
                    now_next_qtf = [now_next_qa, now_next_qa + question_num]

            for similar_user_id in range(similar_user):
                while (len(node_tmp.children.keys()) == 0 or node_tmp.dep > similar_user):
                    node_tmp = node_tmp.fail 

                idx_all.append(node_tmp.idx)

                # [similar_user_num, future_sequence_len, 2]
                result.append([])
                weight.append([])
                P.append([])
                dep_list.append([])
                for future_idx in range(future_sequence):
                    if (now_next_qa == -1):
                        result[-1].append(np.array([0.0, 0.0]))
                        weight[-1].append(np.array([0.0, 0.0]))
                        P[-1].append(np.array([0.0,  0.0]))
                        dep_list[-1].append(np.array([node_tmp.dep + 1 + future_idx, node_tmp.dep + 1 + future_idx]))
                    else:
                        # [2]
                        tmp_result = []
                        tmp_weight = []
                        tmp_P = []
                        tmp_dep_list = []
                        for now_next_qtf_id in now_next_qtf:
                            tmp_result.append(str(now_next_qtf_id))
                            tmp_dep_list.append(node_tmp.dep + 1.0 + future_idx)
                            if ((future_idx + 1) in node_tmp.sample_weight_list.keys() and str(now_next_qtf_id) in node_tmp.sample_weight_list[future_idx + 1].keys()):
                                tmp_weight.append(node_tmp.sample_weight_list[future_idx + 1][str(now_next_qtf_id)])
                                tmp_P.append(node_tmp.sample_weight_list[future_idx + 1][str(now_next_qtf_id)]/sum(node_tmp.sample_weight_list[future_idx + 1].values()))
                            else:
                                tmp_weight.append(0)
                                tmp_P.append(0)
                        result[-1].append(np.array(tmp_result))
                        weight[-1].append(np.array(tmp_weight))
                        P[-1].append(np.array(tmp_P))
                        dep_list[-1].append(np.array(tmp_dep_list))


                if (node_tmp.fail != None):
                    node_tmp = node_tmp.fail

            matched = True
            while char not in node.children:
                if node.fail is None:
                    matched = False
                    break
                node = node.fail
            if matched:
                node = node.children[char]  

            similar.append(result)
            weight_all.append(weight)
            p_all.append(P)
            dep_all.append(dep_list)
            node_idx_all.append(idx_all)

            pre_char[0] = pre_char[1]
            pre_char[1] = char

        return similar,weight_all,p_all, dep_all, node_idx_all

def get_LearningPatternTrie(Source_Sequence, params):
    num_followup_attempts = params.num_followup_attempts
    num_learningpattern = params.num_learningpattern
    print("Create Learning Pattern Trie")
    LPTree = LearningPatternTrie()
    # Source_Sequence: [data_num, seq_len]
    for i in tqdm(range(len(Source_Sequence))):
        for j in range(len(Source_Sequence[0])):
            a = []
            for k in range(j,len(Source_Sequence[0])):
                a.append(str(Source_Sequence[i][k]))
            LPTree.add(a, num_followup_attempts, num_learningpattern)
    print("Complete the Learning Pattern Trie construction!")

    print("Create Fail Link")
    LPTree.finalize()  
    print("Create Fail Link Finished!")

    print("Get Sample Information Begin!")
    LPTree.get_sample_info(LPTree.root, num_followup_attempts)
    print("Get Sample Information End!")

    return LPTree

def load_followup_trend(LPTree, Target_user_Sequence, num_learningpattern, num_followup_attempts, question_num):

    weight_list  = []  # [target_dataset_num, sequence_len, num_learningpattern, num_followup_attempts_weight]
    p_list       = []  # [target_dataset_num, sequence_len, num_learningpattern, num_followup_attempts_probability]
    dep_list     = []  # [target_dataset_num, sequence_len, num_learningpattern, num_followup_attempts_dep]
    idx_list     = []  # [target_dataset_num, sequence_len, num_learningpattern, num_followup_attempts_dep]

    # Target_user_Sequence: [data_num, seqlen]
    for i in range(len(Target_user_Sequence)):
        # if (i % 100 == 0):
        #     print("right now :" + str(i))
        # output_user,output_weight,output_p = AC.search_in_2(Target_user_Sequence[i],num_learningpattern,num_followup_attempts,question_num,probability)
        # output_user,output_weight,output_p, output_dep = AC.search_in_3(Target_user_Sequence[i],num_learningpattern,num_followup_attempts,question_num,probability)
        output_user, output_weight, output_p, output_dep, output_idx = \
            LPTree.search_in_4(Target_user_Sequence[i], num_learningpattern, num_followup_attempts,question_num)
        weight_list.append(output_weight)
        p_list.append(output_p)
        dep_list.append(output_dep)
        idx_list.append(output_idx)

    return np.array(weight_list).astype(np.int32), np.array(p_list), np.array(dep_list).astype(np.int32), np.array(idx_list).astype(np.int32)
