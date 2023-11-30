import numpy as np
import torch
from torch import nn


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

class COTA(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(COTA, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.temperature= args.temperature
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()
        self.nce_fct = nn.CrossEntropyLoss(reduction="sum")
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)


        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)
            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)
            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    '''
    sequence representation encoder module proposed by SASRec
    '''
    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim
        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)
        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)
        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training        
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        return log_feats, pos_logits, neg_logits # pos_pred, neg_pred

    '''
    contrastive learning loss with triplets
    - input:
        - augmented sequence vector with item id
    - output:
        - calculated contrastive learning loss
    '''
    def info_nce_3(self, aug1_seuqence_vectors, aug2_seuqence_vectors, aug3_seuqence_vectors, batch_size):
        N = 3 * batch_size
        z = torch.cat((aug1_seuqence_vectors, aug2_seuqence_vectors, aug3_seuqence_vectors), dim=0)
        cos = nn.CosineSimilarity(dim=2, eps=1e-8)
        sim = cos(z.unsqueeze(1), z.unsqueeze(0))
        sim = sim / self.temperature
        sim_i_j1= torch.diag(sim, batch_size)
        sim_i_j2 = torch.diag(sim, 2*batch_size)
        sim_j_i1 = torch.diag(sim, -batch_size)
        sim_j_i2 = torch.diag(sim, -2*batch_size)
        positive_samples = torch.cat((sim_i_j1, sim_j_i2, sim_i_j2, sim_j_i1), dim=0).reshape(2,N).T
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size*2 + i] = 0
            mask[batch_size*2 + i, i] = 0
        for i in range(2*batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        negative_samples = sim[mask]
        negative_samples=negative_samples.reshape(N,-1)
        labels = torch.zeros(N).to(positive_samples.device).long()
        added_positive_samples=positive_samples[:,0]+positive_samples[:,1]
        logits = torch.cat((added_positive_samples.reshape(N,1), negative_samples), dim=1)
        loss=self.nce_fct(logits, labels)
        loss/=N
        return loss
    '''
    calculation of final loss
    - input:
        - au1, aug2, aug3: representation of augmented sequences
        - log_feats, pos_logits,pos_labels,neg_logits: for original sequence
    - output: 
        - final loss = main loss of base model + weight * contrastive learning loss of COTA
    '''
    def cal_loss_3(self,log_feats, pos_logits,pos_labels,neg_logits,neg_labels,indices, aug1, aug2, aug3, args):
        # recommendation loss
        rec_loss = self.bce_criterion(pos_logits[indices], pos_labels[indices])
        rec_loss += self.bce_criterion(neg_logits[indices], neg_labels[indices])
        # contrastive learning loss
        aug1_log_feats=self.log2feats(aug1)
        aug2_log_feats=self.log2feats(aug2)
        aug3_log_feats=self.log2feats(aug3)
        cl_loss=self.info_nce_3(aug1_log_feats[:, -1, :], aug2_log_feats[:,-1,:],aug3_log_feats[:,-1,:], args.batch_size)
        return rec_loss + args.cl_weight*cl_loss
    
    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet
        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits # preds # (U, I)
