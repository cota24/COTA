import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict, Counter
from multiprocessing import Process, Queue
from scipy.stats import entropy
from tqdm import tqdm

'''
Cluster-based Sequence Augmentation: Head-to-Tail Replacement
- input: original sequence, augmentation proportion, head item set
- output: augmented sequence
'''
def general_masking(seq, general_masking_proportion, generalset):
    for j in range(len(seq)):
        maksing_iid = random.sample(
            set(seq[j]) & generalset,
            int(len(set(seq[j]) & generalset) * general_masking_proportion),
        )
        for i in range(len(seq[j])):
            if seq[j][i] in maksing_iid:
                seq[j][i] = 0

    return seq

'''
Cluster-based Sequence Augmentation: Head-to-Tail Replacement
- input: original sequence, augmentation proportion, cluster information dictionary(included items), cluster information dictionary(included tail items), tail item set
- output: augmented sequence
'''
def cate_based_item_changing(seq, change_proportion, cluster_iid_dict, cluster_taillist_dict, np_tail):
    for j in range(len(seq)):
        changing_indices = np.random.choice(len(seq[j]), int(len(seq[j]) * change_proportion), replace=False)
        for idx in changing_indices:
            if seq[j][idx] in np_tail or seq[j][idx] == 0:
                continue
            tar_clusterid = cluster_iid_dict.get(seq[j][idx], None)
            if tar_clusterid is not None:
                tail_cluster_items = cluster_taillist_dict.get(tar_clusterid, []).item()
                if len(tail_cluster_items) > 0:
                    seq[j][idx] = random.choice(list(tail_cluster_items))
    return seq

'''
Cluster-based Sequence Augmentation: Tail Insertion
- input: original sequence, augmentation proportion, cluster information dictionary(included items), cluster information dictionary(included tail items), tail item set, maximum length of sequence
- output: augmented sequence
'''
def tail_insertion(seq, insert_proportion, cluster_iid_dict, cluster_taillist_dict, tailset, maxlen):
    for j in range(len(seq)):
        num_inserted = int(len(seq[j]) * insert_proportion)
        changing_indices = np.random.choice(len(seq[j]), num_inserted, replace=False)
        for changing_index in changing_indices:
            if seq[j][changing_index] in tailset or seq[j][changing_index] == 0:
                continue
            tar_clusterid = cluster_iid_dict.get(seq[j][changing_index], None)
            if tar_clusterid is not None:
                tail_cluster_items = cluster_taillist_dict.get(tar_clusterid, []).item()
                if len(tail_cluster_items) > 0:
                    insert_itemid = random.choice(list(tail_cluster_items))
                    seq[j] = np.concatenate((seq[j][:changing_index+1], [insert_itemid], seq[j][changing_index+1:]))[:maxlen]
    return seq

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))

class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s/%s.txt' % (fname,fname), 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            # user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            # user_test[user] = []
            # user_test[user].append(User[user][-1])
    return [user_train, user_valid, usernum, itemnum] # user_test

'''
evaluation of nDCG, Hit Rate, coverage, tail coverage, Entropy
model return top k recommendation list (k=10)
'''
def evaluate(model, dataset, args, tailset):
    [train, valid, usernum, itemnum] = copy.deepcopy(dataset)
    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    total_item = []
    selected_tail_dup=[]
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1:
            continue
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        unrated= set(range(1, itemnum+1)) - set(train[u]) - set([valid[u][0]])
        item_idx = list(unrated) + [valid[u][0]]
        predictions = model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]
        _, topk = torch.topk(predictions, 10)
        topk = np.array(item_idx)[topk.cpu()]
        if valid[u][0] in topk:
            HT += 1
            rank = (topk==valid[u][0]).nonzero()[0]
            NDCG += 1 / np.log2(rank + 2)
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()  
        total_item.extend(topk)
        valid_user += 1

    coverage = len(Counter(total_item).keys()) / itemnum
    tailcoverage=len(set(Counter(total_item).keys()) & tailset)/len(tailset)
    frequency_rec=np.array(list(Counter(total_item).values()))
    recommended_prop=frequency_rec/len(total_item)
    H = entropy(recommended_prop, base=2)
    return NDCG / valid_user, HT / valid_user, coverage, tailcoverage, H

