import os
import time
import torch
import itertools
import argparse
import pandas as pd
from tqdm import tqdm
import pickle
from model import COTA
from utils import *

def str2bool(s):
    if s not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return s == "true"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--train_dir", required=True)
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--maxlen", default=50, type=int) 
parser.add_argument("--hidden_units", default=50, type=int)
parser.add_argument("--num_blocks", default=2, type=int)
parser.add_argument("--num_epochs", default=200, type=int)
parser.add_argument("--num_heads", default=2, type=int)
parser.add_argument("--dropout_rate", default=0.2, type=float)
parser.add_argument("--l2_emb", default=0.0, type=float)
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--inference_only", default=0, type=int)
parser.add_argument("--state_dict_path", default=None, type=str)
parser.add_argument("--cl_weight", default=0.001, type=float) 
parser.add_argument("--general_masking_proportion", default=0.3, type=float) 
parser.add_argument("--weak_change_proportion", default=0.1, type=float) 
parser.add_argument("--strong_change_proportion", default=0.3, type=float) 
parser.add_argument("--insert_proportion", default=0.1, type=float) 
parser.add_argument("--temperature", default=1, type=float)

args = parser.parse_args()
if not os.path.isdir(args.dataset + "_" + args.train_dir):
    os.makedirs(args.dataset + "_" + args.train_dir)
with open(os.path.join(args.dataset + "_" + args.train_dir, "args.txt"), "w") as f:
    f.write(
        "\n".join(
            [
                str(k) + "," + str(v)
                for k, v in sorted(vars(args).items(), key=lambda x: x[0])
            ]
        )
    )
f.close()

if __name__ == "__main__":
    dataset = data_partition(args.dataset)
    [user_train, user_valid, usernum, itemnum] = dataset # user_test
    num_batch = (
        len(user_train) // args.batch_size
    )  
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print("average sequence length: %.2f" % (cc / len(user_train)))

    '''
    tailset: set of tail item ids
    generalset: set of head item ids
    
    '''
    with open("data/"+args.dataset+"/"+args.dataset+"_tail.pkl","rb") as f:
        np_tail=pickle.load(f) 
    tailset=set(np_tail) 
    with open("data/"+args.dataset+"/"+args.dataset+"_head.pkl","rb") as f:
        np_general=pickle.load(f)
    generalset=set(np_general)
    with open("data/"+args.dataset+"/"+args.dataset+"_cl_iid.pkl","rb") as f:
        np_cluster_iid=pickle.load(f)
    with open("data/"+args.dataset+"/"+args.dataset+"_cl_tlist.pkl","rb") as f:
        np_cluster_taillist=pickle.load(f)
    cluster_iid_dict = {row[1]: row[0] for row in np_cluster_iid}
    cluster_taillist_dict = {row[0]: row[1] for row in np_cluster_taillist}
    
    f = open(os.path.join(args.dataset + "_" + args.train_dir, "log.txt"), "w")
    sampler = WarpSampler(
        user_train,
        usernum,
        itemnum,
        batch_size=args.batch_size,
        maxlen=args.maxlen,
        n_workers=3,
    )
    model = COTA(usernum, itemnum, args).to(
        args.device
    ) 
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass  
    model.train()  
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        model.load_state_dict(
            torch.load(args.state_dict_path, map_location=torch.device(args.device))
        )
        tail = args.state_dict_path[args.state_dict_path.find("epoch=") + 6 :]
        epoch_start_idx = int(tail[: tail.find(".")]) + 1

    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args, tailset)
        print(
            "test (NDCG@10: %.4f, HR@10: %.4f), Coverage@10: %.4f, Tail Coverage@10:%.4f, Tail Ratio@10:%.4f, Entropy@10:%.4f"
            % (t_test[0], t_test[1], t_test[2], t_test[3], t_test[4], t_test[5])
        )

    bce_criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    T = 0.0
    t0 = time.time()
    for epoch in tqdm(range(epoch_start_idx, args.num_epochs + 1)):
        if args.inference_only:
            break  # just to decrease identition
        for step in tqdm(
            range(num_batch), total=num_batch, ncols=70, leave=False, unit="b"
        ):  # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch()  # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            # augmenation fro cl loss
            aug1 = general_masking(
                seq.copy(), args.general_masking_proportion, generalset
            )
            aug2 = cate_based_item_changing(
                seq.copy(),
                args.weak_change_proportion,
                cluster_iid_dict,
                cluster_taillist_dict,
                np_tail
            )
            aug3 = cate_based_item_changing(
                seq.copy(),
                args.strong_change_proportion,
                cluster_iid_dict,
                cluster_taillist_dict,
                np_tail
            )
            # data increase
            aug_num=random.choices(range(1, 5), weights = [0.05, 0.1, 0.1, 0.75])
            if aug_num == 1:
                seq = tail_insertion(
                    seq.copy(),
                    args.insert_proportion,
                    cluster_iid_dict,
                    cluster_taillist_dict,
                    np_tail,
                    args.maxlen
                )
            elif aug_num == 2:
                seq = general_masking(
                    seq.copy(), args.general_masking_proportion, generalset
                )
            elif aug_num == 3:
                seq = cate_based_item_changing(
                    seq.copy(),
                    args.strong_change_proportion,
                    cluster_iid_dict,
                    cluster_taillist_dict,
                    np_tail
                )
            log_feats, pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(
                pos_logits.shape, device=args.device
            ), torch.zeros(neg_logits.shape, device=args.device)
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            
            # cl loss + main loss
            loss = model.cal_loss_3(
                log_feats,
                pos_logits,
                pos_labels,
                neg_logits,
                neg_labels,
                indices,
                aug1,
                aug2,
                aug3,
                args,
            )

            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()

        if epoch == args.num_epochs: 
            model.eval()
            t1 = time.time() - t0
            T += t1
            print("Evaluating", end="")
            t_valid = evaluate(model, dataset, args, tailset)
            print(
                "epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f, Coverage@10: %.4f, Tail Coverage@10:%.4f, Entropy@10:%.4f)"
                % (
                    epoch,
                    T,
                    t_valid[0],
                    t_valid[1],
                    t_valid[2],
                    t_valid[3],
                    t_valid[4]
                )
            )

            f.write(str(t_valid) + "\n")
            f.flush()
            t0 = time.time()
            model.train()
            folder = args.dataset + "_" + args.train_dir
            fname = "COTA.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth"
            fname = fname.format(
                args.num_epochs,
                args.lr,
                args.num_blocks,
                args.num_heads,
                args.hidden_units,
                args.maxlen,
            )
            torch.save(model.state_dict(), os.path.join(folder, fname))
    f.close()
    sampler.close()
    print("Done")
    
    
        

