import lstm_crf
import argparse
from my_utils import *
import os
use_cuda = False
rnd_seed = 1

parser = argparse.ArgumentParser()
parser.add_argument("-tr", "--train_file", type=str,
                default="../ner_data/eng_data/eng.train.true.conll")
parser.add_argument("-d", "--dev_file", type=str,
                default="../ner_data/eng_data/eng.testa.true.conll")
parser.add_argument("-te", "--test_file", type=str,
                    default="../ner_data/eng_data/eng.testb.true.conll")
# parser.add_argument("-e", "--embedding_file", type=str,
#                     default="../embeddings/glove.840B.300d.txt")
parser.add_argument("-e", "--embedding_file", type=str,
                    default="../embeddings/glove_dummy.txt")
parser.add_argument("-o", "--output_file", type=str,
                    default="./annotations/output_resumed_again.conll")
parser.add_argument("-m", "--model_name", type=str,
                    default="./models/trial_before_wsd.mod")
parser.add_argument("-p", "--pretrained_model_name", type=str,
                    default="./models/ner_lstm_crf_resumed.mod.4")
parser.add_argument("--use_cuda", dest="use_cuda", action='store_true')

args = parser.parse_args()
if args.use_cuda:
    use_cuda = True

train_file = args.train_file
dev_file = args.dev_file
test_file = args.test_file
embedding_file = args.embedding_file

import torch.autograd as A, torch.optim as O
if not use_cuda:
    import torch
else:
    import torch.cuda as torch # This must be done after importing all other modules within torch since torch.cuda does
    # NOT have overridden versions of autograd, nn etc.
torch.manual_seed(rnd_seed)


tag_identifier = torch.load(args.pretrained_model_name+"_tag_identifier") if os.path.isfile(args.pretrained_model_name+"_tag_identifier") else None
char_identifier = torch.load(args.pretrained_model_name+"_char_identifier") if os.path.isfile(args.pretrained_model_name+"_char_identifier") else None
word_identifier = torch.load(args.pretrained_model_name+"_word_identifier") if os.path.isfile(args.pretrained_model_name+"_word_identifier") else None

train_instances, dev_instances, test_instances, word_embedder, word_identifier, tag_identifier, char_identifier = \
    get_ner_samples(train_file, dev_file, test_file, embedding_file, tag_identifier=tag_identifier,
                    word_identifier=word_identifier, char_identifier=char_identifier, use_cuda=use_cuda,
                    use_pretrained=not (None in [tag_identifier, char_identifier, word_identifier]))

torch.save(tag_identifier, args.model_name + "_tag_identifier")
torch.save(word_identifier, args.model_name + "_word_identifier")
torch.save(char_identifier, args.model_name + "_char_identifier")

wfeat_dim = word_embedder.wemb_dim
cfeat_dim = 1

n_epochs = 25
batch_size = 1
wlstm_layers = 1
wlstm_dim = 100
wemb_dim = 32

clstm_layers = 1
clstm_dim = 16
cemb_dim = 16

crf_ip_dim = 64

n_tags = len(tag_identifier.wid2word)

neural_crf = lstm_crf.Neural_CRF(wvocab_size=len(word_identifier.wid2word)+1, wlstm_layers=wlstm_layers,
                                 wlstm_dim=wlstm_dim,
                                 wfeat_dim=wfeat_dim,
                                 wemb_dim=wemb_dim, cvocab_size=len(char_identifier.wid2word)+1,
                                 clstm_layers=clstm_layers, clstm_dim=clstm_dim, cfeat_dim=cfeat_dim,
                                 cemb_dim=cemb_dim, crf_ip_dim=crf_ip_dim, n_tags=n_tags)
# Note: When specifying the vocabulary size, you must treat UNK as a word as well, even though it is masked by 0s. If
# this is not done, you will end up errors whenever a word whose index corresponds the the last vocabulary item is used
if os.path.isfile(args.pretrained_model_name):
    neural_crf.load_state_dict(torch.load(args.pretrained_model_name))

optimizer = O.SGD(params=neural_crf.parameters(),lr=0.01,momentum=0.0,dampening=0,weight_decay=0.0,nesterov=False)

train_instances += dev_instances

print("Starting training procedure!")

sample_ids = range(len(train_instances))
for epoch in range(n_epochs):
    batch_loss = A.Variable(torch.zeros(1))
    epoch_loss = 0.0
    np.random.shuffle(sample_ids) # Uncomment this eventually!
    for order_id, sid in enumerate(sample_ids):
        train_instance = train_instances[sid]
        crf_loss = neural_crf.infer_forward(neural_crf(wids=train_instance.wids, wfeats=train_instance.wfeats, cids=train_instance.cids,
                                                       cfeats=train_instance.cfeats), label_ids=train_instance.tags,
                                                       label_identifier=tag_identifier)
        batch_loss += crf_loss
        epoch_loss += crf_loss.data[0]
        # print("Processed %d sentences\r"%(order_id)),
        print("Processed %d sentences" % (order_id))
        if (order_id+1)%batch_size==0 or order_id==len(sample_ids)-1:
            neural_crf.zero_grad()
            batch_loss.backward()
            optimizer.step()
            # try:
            #     print("\nTotal loss for batch within epoch %d = %f" % (epoch+1, batch_loss.data[0]))
            # except:
            #     print("\nCouldn't print loss!")
            batch_loss = A.Variable(torch.zeros(1))
    print("\nAverage loss for epoch %d = %f"%(epoch+1, epoch_loss/len(sample_ids)))
    annotate_corpus(args.output_file + "." + str(epoch+1), test_instances, neural_crf, tag_identifier)
    torch.save(neural_crf.state_dict(), args.model_name+"."+str(epoch+1))


print("DONE!")

# NOTE THAT A USEFUL FOLLOW UP TO THIS IS TO IMPLEMENT A STRUCTURED PERCEPTRON LOSS! THIS IS THE DIFFERENCE BETWEEN THE
# VITERBI PATH SCORE AND THE GOLD STANDARD PATH SCORE