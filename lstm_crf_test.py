import lstm_crf

use_cuda = False
rnd_seed = 1
import torch.autograd as A, torch.optim as O

if not use_cuda:
    import torch
else:
    import torch.cuda as torch # This must be done after importing all other modules within torch since torch.cuda does
    # NOT have overridden versions of autograd, nn etc.

torch.manual_seed(rnd_seed)

wfeat_dim = 5
cfeat_dim = 5

# dummy data

wvocab = ['w1','w2','w3','w4','w5','w6','w19']
cvocab = set(list("".join(wvocab)))

w2id = {w:i for (i, w) in enumerate(wvocab)}
c2id = {c:i for (i, c) in enumerate(cvocab)}

wfeat_dict = {w:torch.rand(1, wfeat_dim) for w in wvocab}
cfeat_dict = {c:torch.rand(1,cfeat_dim) for c in cvocab}

sentences = [
             ['w1','w2','w3','w4'],
             ['w5','w6','w1','w2','w3'],
             ['w4','w5','w6'],
             ['w1','w19']
            ]

# The dummy data corresponds to NER tasks

# 1=B-PER,  2=I-PER,  3=O,  4=B-LOC,  5=I-LOC
tags = [
        [0, 3, 1, 2, 3, 6],
        [0, 4, 5, 3, 1, 2, 6],
        [0, 3, 4, 5, 6],
        [0, 3, 1, 6]
       ]

wids = [A.Variable(torch.LongTensor([w2id[w] for w in wlist])).unsqueeze(0) for wlist in sentences]
wfeats = [
            A.Variable(torch.cat(
                [
                    wfeat_dict[w]
                for w in wlist
                ],0)
            ).unsqueeze(0)
         for wlist in sentences
         ]
cids = [
            [
                A.Variable(torch.LongTensor(
                [
                 c2id[c]
                for c in w
                ])).unsqueeze(0)
            for w in wlist
            ]
       for wlist in sentences
       ]

cfeats = [
    [
        A.Variable(torch.cat(
            [
                cfeat_dict[c]
                for c in w
            ],0)).unsqueeze(0)
        for w in wlist
    ]
    for wlist in sentences
]

n_epochs = 25
wlstm_layers = 2
wlstm_dim = 32
wemb_dim = 64

clstm_layers = 2
clstm_dim = 16
cemb_dim = 24

crf_ip_dim = 16

n_tags = 5

neural_crf = lstm_crf.Neural_CRF(wvocab_size=len(wvocab), wlstm_layers=2, wlstm_dim=wlstm_dim, wfeat_dim=wfeat_dim,
                                 wemb_dim=wemb_dim, cvocab_size=len(cvocab), clstm_layers=clstm_dim,
                                 clstm_dim=clstm_dim, cfeat_dim=cfeat_dim,
                                 cemb_dim=cemb_dim, crf_ip_dim=16, n_tags=n_tags)

optimizer = O.RMSprop(params=neural_crf.parameters(), lr=0.01, weight_decay=0.001,momentum=0.001)

for epoch in range(n_epochs):
    tot_loss = A.Variable(torch.zeros(1))
    for sid in range(len(sentences)):
        crf_loss = neural_crf.infer_forward(neural_crf(wids=wids[sid], wfeats=wfeats[sid], cids=cids[sid],
                                                       cfeats=cfeats[sid]), label_ids=tags[sid])
        tot_loss += crf_loss
        neural_crf.zero_grad()
        crf_loss.backward()
        optimizer.step()
    print("Epoch Loss for epoch %d = %f"%(epoch, tot_loss.data[0]/3))

for sid in range(len(sentences)):
    pred_labels, final_tag_score = neural_crf.annotate(neural_crf(wids=wids[sid], wfeats=wfeats[sid], cids=cids[sid],
                                   cfeats=cfeats[sid]))
    # compare label_ids=tags[sid] and pred_labels
    print("Predicted labels ",pred_labels)
    print("Gold labels ",tags[sid])
    print("\n\n")

print("DONE!")

# NOTE THAT A USEFUL FOLLOW UP TO THIS IS TO IMPLEMENT A STRUCTURED PERCEPTRON LOSS! THIS IS THE DIFFERENCE BETWEEN THE
# VITERBI PATH SCORE AND THE GOLD STANDARD PATH SCORE