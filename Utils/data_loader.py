from Utils.LearningPatternTrie import load_followup_trend, get_LearningPatternTrie
from Utils.KMP import load_followup_trend_KMP
import numpy as np
import math, time, psutil

class KTDATA(object):
    def __init__(self, n_question, seqlen, separate_char, name="data"):
        # In the ASSISTments2009 dataset:
        # param: n_queation = 110
        #        seqlen = 200
        self.separate_char = separate_char
        self.n_question = n_question
        """
        self.seqlen = seqlen+1
        """
        self.seqlen = seqlen

    ### data format
    ### 15
    ### 1,1,1,1,7,7,9,10,10,10,10,11,11,45,54
    ### 0,1,1,1,1,1,0,0,1,1,1,1,1,0,0
    def load_data(self, path):
        f_data = open(path , 'r')
        q_data = []
        qa_data = []
        for lineID, line in enumerate(f_data):
            line = line.strip( )
            # lineID starts from 0
            if lineID % 3 == 1:
                Q = line.split(self.separate_char)
                if len( Q[len(Q)-1] ) == 0:
                    Q = Q[:-1]
                #print(len(Q))
            elif lineID % 3 == 2:
                A = line.split(self.separate_char)
                if len( A[len(A)-1] ) == 0:
                    A = A[:-1]
                #print(len(A),A)

                # start split the data
                n_split = 1
                #print('len(Q):',len(Q))
                if len(Q) > self.seqlen:
                    n_split = math.floor(len(Q) / self.seqlen)
                    if len(Q) % self.seqlen:
                        n_split = n_split + 1
                #print('n_split:',n_split)
                for k in range(n_split):
                    question_sequence = []
                    answer_sequence = []
                    if k == n_split - 1:
                        endINdex  = len(A)
                    else:
                        endINdex = (k+1) * self.seqlen
                    for i in range(k * self.seqlen, endINdex):
                        if len(Q[i]) > 0 :
                            # int(A[i]) is in {0,1}
                            Xindex = int(Q[i]) + int(A[i]) * self.n_question
                            question_sequence.append(int(Q[i]))
                            answer_sequence.append(Xindex)
                        else:
                            print(Q[i])
                    #print('instance:-->', len(instance),instance)
                    q_data.append(question_sequence)
                    qa_data.append(answer_sequence)
        f_data.close()
        ### data: [[],[],[],...] <-- set_max_seqlen is used
        ### convert data into ndarrays for better speed during training
        q_dataArray = np.zeros((len(q_data), self.seqlen))
        for j in range(len(q_data)):
            dat = q_data[j]
            q_dataArray[j, :len(dat)] = dat

        qa_dataArray = np.zeros((len(qa_data), self.seqlen))
        for j in range(len(qa_data)):
            dat = qa_data[j]
            qa_dataArray[j, :len(dat)] = dat
        # dataArray: [ array([[],[],..])] Shape: (3633, 200)
        return q_dataArray, qa_dataArray

    def generate_all_index_data(self, batch_size):
        n_question = self.n_question
        batch = math.floor( n_question / self.seqlen )
        if self.n_question % self.seqlen:
            batch += 1

        seq = np.arange(1, self.seqlen * batch + 1)
        zeroindex = np.arange(n_question, self.seqlen * batch)
        zeroindex = zeroindex.tolist()
        seq[zeroindex] = 0
        q = seq.reshape((batch,self.seqlen))
        q_dataArray = np.zeros((batch_size, self.seqlen))
        q_dataArray[0:batch, :] = q
        return q_dataArray

def load_DataSet(params, logs, search_type = "Trie"):
    #region Step 1: Load Raw Data
    dat = KTDATA(n_question=params.n_question, seqlen=params.seqlen, separate_char=',')
    print(params.data_dir + "/" + params.data_name + "_train" + str(params.seed) + ".csv")
    train_data_path = params.data_dir + "/" + params.data_name + "_train" + str(params.seed) + ".csv"
    valid_data_path = params.data_dir + "/" + params.data_name + "_valid" + str(params.seed) + ".csv"
    test_data_path = params.data_dir + "/" + params.data_name + "_test.csv"

    train_q_data, train_qa_data = dat.load_data(train_data_path)
    valid_q_data, valid_qa_data = dat.load_data(valid_data_path)
    test_q_data, test_qa_data = dat.load_data(test_data_path)

    if params.num_train_test_valid != -1:
        train_q_data, train_qa_data = train_q_data[:params.num_train_test_valid], train_qa_data[:params.num_train_test_valid]
        valid_q_data, valid_qa_data = valid_q_data[:params.num_train_test_valid], valid_qa_data[:params.num_train_test_valid]
        test_q_data, test_qa_data = test_q_data[:params.num_train_test_valid], test_qa_data[:params.num_train_test_valid]
    
    all_dataset_list = [train_q_data, train_qa_data, valid_q_data, valid_qa_data, test_q_data, test_qa_data]
    train_q_data, train_qa_data, valid_q_data, valid_qa_data, test_q_data, test_qa_data = [data[:int(len(data) * params.datascalability)] for data in all_dataset_list]

    print('train_qa_data_shape: ', train_qa_data.shape)
    print('valid_qa_data_shape: ', valid_qa_data.shape)
    print('test_qa_data_shape: ', test_qa_data.shape)
    DataSet = {
        "train":[train_q_data, train_qa_data],
        "valid":[valid_q_data, valid_qa_data],
        "test":[test_q_data, test_qa_data]
    }
    #endregion

    #region Step 2: Generate or Load Followup Trends if needed
    LPTree = FollowupTrends = {}
    if "FINER" in params.model_type:
        if (params.datascalability == 1):
            LPTree_file_name = './Results/FollowupTrends/' + params.dataset_type + params.date + '_' + str(
                params.num_learningpattern) + '_' + str(params.num_followup_attempts) + '_' + str(params.seed)
        else:
            LPTree_file_name = './Results/FollowupTrends/' + params.dataset_type + "_" + str(params.datascalability) + "_" + params.date + '_' + str(
                params.num_learningpattern) + '_' + str(params.num_followup_attempts) + '_' + str(params.seed)
        if params.generate_followup_trends:
            if (search_type == "Trie"):
                logs.print("Begin Build Learning Pattern Trie !!!")
                start_time = time.time()
                
                # Record memory usage before creating LPTree
                memory_before = psutil.Process().memory_info().rss / (1024 * 1024)
                LPTree = get_LearningPatternTrie(train_qa_data, params)
                # Record memory usage after creating LPTree
                memory_after = psutil.Process().memory_info().rss / (1024 * 1024)
                # Calculate the memory usage of LPTree
                lptree_memory_usage = memory_after - memory_before

                logs.print("The time cost of building learning pattern trie: " + str(time.time() - start_time) + "s.")
                logs.print(f"The memory cost of LPTree: {lptree_memory_usage:.2f} MB")
                start_time = time.time()
                weight_train, p_train, dep_train, idx_train = load_followup_trend(LPTree,
                                                                                train_qa_data,
                                                                                params.num_learningpattern,
                                                                                params.num_followup_attempts,
                                                                                params.n_question)
                logs.print("The time cost of getting followup trends of training data: " + str((time.time() - start_time) / params.batch_size) + "s.")
                start_time = time.time()
                weight_valid, p_valid, dep_valid, idx_valid = load_followup_trend(LPTree,
                                                                                valid_qa_data,
                                                                                params.num_learningpattern,
                                                                                params.num_followup_attempts,
                                                                                params.n_question)
                logs.print("The time cost of getting followup trends of valid data: " + str((time.time() - start_time) / params.batch_size) + "s.")
                start_time = time.time()
                weight_test, p_test, dep_test, idx_test = load_followup_trend(LPTree,
                                                                            test_qa_data,
                                                                            params.num_learningpattern,
                                                                            params.num_followup_attempts,
                                                                            params.n_question)
                logs.print("The time cost of getting followup trends of test data: " + str((time.time() - start_time) / params.batch_size) + "s.")
                np.savez(LPTree_file_name + '_weight.npz',
                        weight_train=weight_train,
                        weight_vaild=weight_valid,
                        weight_test=weight_test)
                np.savez(LPTree_file_name + '_p.npz',
                        p_train=p_train,
                        p_vaild=p_valid,
                        p_test=p_test)
                np.savez(LPTree_file_name + '_dep.npz',
                        dep_train=dep_train,
                        dep_vaild=dep_valid,
                        dep_test=dep_test)
                np.savez(LPTree_file_name + '_idx.npz',
                        idx_train=idx_train,
                        idx_vaild=idx_valid,
                        idx_test=idx_test)
                print("The memory cost of building learning pattern trie: ", ".")
            else:
                logs.print("Begin Get FPTs by utilizing KMP !!!")
                start_time = time.time()
                weight_train, p_train, dep_train, idx_train = load_followup_trend_KMP(train_qa_data,
                                                                                train_qa_data,
                                                                                params.num_learningpattern,
                                                                                params.num_followup_attempts,
                                                                                params.n_question)
                logs.print("The time cost of KMP getting followup trends of training data: " + str((time.time() - start_time) / params.batch_size) + "s.")
                start_time = time.time()
                weight_valid, p_valid, dep_valid, idx_valid = load_followup_trend_KMP(train_qa_data,
                                                                                valid_qa_data,
                                                                                params.num_learningpattern,
                                                                                params.num_followup_attempts,
                                                                                params.n_question)
                logs.print("The time cost of KMP getting followup trends of valid data: " + str((time.time() - start_time) / params.batch_size) + "s.")
                start_time = time.time()
                weight_test, p_test, dep_test, idx_test = load_followup_trend_KMP(train_qa_data,
                                                                            test_qa_data,
                                                                            params.num_learningpattern,
                                                                            params.num_followup_attempts,
                                                                            params.n_question)
                logs.print("The time cost of KMP getting followup trends of test data: " + str((time.time() - start_time) / params.batch_size) + "s.")
        else:
            print("load frequency of FPTs!")
            arch = np.load(LPTree_file_name + '_weight.npz')
            weight_train = arch['weight_train']
            weight_valid = arch['weight_vaild']
            weight_test = arch['weight_test']

            print("load probabilisty of FPTs!")
            arch = np.load(LPTree_file_name + '_p.npz')
            p_train = arch['p_train']
            p_valid = arch['p_vaild']
            p_test = arch['p_test']

            print("load length of FPTs!")
            arch = np.load(LPTree_file_name + '_dep.npz')
            dep_train = arch['dep_train']
            dep_valid = arch['dep_vaild']
            dep_test = arch['dep_test']

            print("load idx of FPTs!")
            arch = np.load(LPTree_file_name + '_idx.npz')
            idx_train = arch['idx_train']
            idx_valid = arch['idx_vaild']
            idx_test = arch['idx_test']
        FollowupTrends = {
            "train":[weight_train, p_train, dep_train, idx_train],
            "valid":[weight_valid, p_valid, dep_valid, idx_valid],
            "test": [weight_test,  p_test,  dep_test,  idx_test ],
        }
    #endregion
    return DataSet, FollowupTrends

def get_hyper_parameters_dataset(parser):
    parser.add_argument('--dataset_type', type=str, default="assist2009",
                        help='the type of dataset, e.g. assist2009, assist2012, assist2015, HDUOJ, Junyi, Algebra08')
    parser.add_argument('--datascalability', type=float, default=1, help='The scalability of dataset')
    parser.add_argument('--num_train_test_valid', type=int, default=-1, help='the number of training, valid and test data')
 
    params = parser.parse_args()
    if params.dataset_type == 'assist2009':
        parser.add_argument('--n_question', type=int, default=110, help='the number of unique questions in the dataset')
        parser.add_argument('--seqlen', type=int, default=200, help='the allowed maximum length of a sequence')
        parser.add_argument('--data_dir', type=str, default='./Data/assist2009', help='data directory')
        parser.add_argument('--data_name', type=str, default='assist2009', help='data set name')

    elif params.dataset_type == 'assist2015':
        parser.add_argument('--n_question', type=int, default=110, help='the number of unique questions in the dataset')
        parser.add_argument('--seqlen', type=int, default=200, help='the allowed maximum length of a sequence')
        parser.add_argument('--data_dir', type=str, default='./Data/assist2015', help='data directory')
        parser.add_argument('--data_name', type=str, default='assist2015', help='data set name')

    elif params.dataset_type == 'assist2012':
        parser.add_argument('--n_question', type=int, default=198, help='the number of unique questions in the dataset')
        parser.add_argument('--seqlen', type=int, default=200, help='the allowed maximum length of a sequence')
        parser.add_argument('--data_dir', type=str, default='./Data/assist2012', help='data directory')
        parser.add_argument('--data_name', type=str, default='assist2012', help='data set name')

    elif params.dataset_type == 'Algebra08':
        parser.add_argument('--n_question', type=int, default=424, help='the number of unique questions in the dataset')
        parser.add_argument('--seqlen', type=int, default=200, help='the allowed maximum length of a sequence')
        parser.add_argument('--data_dir', type=str, default='./Data/Algebra08', help='data directory')
        parser.add_argument('--data_name', type=str, default='Algebra08', help='data set name')

    elif params.dataset_type == 'HDUOJ':
        parser.add_argument('--n_question', type=int, default=5320,
                            help='the number of unique questions in the dataset')
        parser.add_argument('--seqlen', type=int, default=100, help='the allowed maximum length of a sequence')
        parser.add_argument('--data_dir', type=str, default='./Data/HDUOJ', help='data directory')
        parser.add_argument('--data_name', type=str, default='HDUOJ', help='data set name')
    
    elif params.dataset_type == 'Junyi':
        parser.add_argument('--n_question', type=int, default=715,
                            help='the number of unique questions in the dataset')
        parser.add_argument('--seqlen', type=int, default=100, help='the allowed maximum length of a sequence')
        parser.add_argument('--data_dir', type=str, default='./Data/Junyi', help='data directory')
        parser.add_argument('--data_name', type=str, default='Junyi', help='data set name')

    return parser