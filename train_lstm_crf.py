import lstm_crf
import argparse
from my_utils import *
import os
rnd_seed = 1

parser = argparse.ArgumentParser()
parser.add_argument("-tr", "--train_file", type=str)
parser.add_argument("-d", "--dev_file", type=str,
                default=None)
parser.add_argument("-te", "--test_file", type=str,
                    default=None)
parser.add_argument("-e", "--embedding_file", type=str)
parser.add_argument("-m", "--model_name", type=str,
                    default="./model.mod")
parser.add_argument("-p", "--pretrained_model_name", type=str,
                    default="./non-existent_model")
parser.add_argument("-n", "--n_epochs", type=int,
                    default=25)
parser.add_argument("-w", "--wlstm_dim", type=int,
                    default=100)
parser.add_argument("-fwed", "--fintetuned_embedding_dim", type=int,
                    default=32)
parser.add_argument("-c", "--clstm_dim", type=int,
                    default=100)
parser.add_argument("-fced", "--fintetuned_cembedding_dim", type=int,
                    default=16)
parser.add_argument("-crf", "--crf_ip_dim", type=int,
                    default=64)


args = parser.parse_args()

train_file = args.train_file
dev_file = args.dev_file
test_file = args.test_file
embedding_file = args.embedding_file

import torch.autograd as A, torch.optim as O
import torch
torch.manual_seed(rnd_seed)


tag_identifier = torch.load(args.pretrained_model_name+"_tag_identifier") if os.path.isfile(args.pretrained_model_name+"_tag_identifier") else None
char_identifier = torch.load(args.pretrained_model_name+"_char_identifier") if os.path.isfile(args.pretrained_model_name+"_char_identifier") else None
word_identifier = torch.load(args.pretrained_model_name+"_word_identifier") if os.path.isfile(args.pretrained_model_name+"_word_identifier") else None

train_instances, dev_instances, test_instances, word_embedder, word_identifier, tag_identifier, char_identifier = \
    get_ner_samples(train_file, dev_file, test_file, embedding_file, tag_identifier=tag_identifier,
                    word_identifier=word_identifier, char_identifier=char_identifier, use_cuda=False,
                    use_pretrained=not (None in [tag_identifier, char_identifier, word_identifier]))

torch.save(tag_identifier, args.model_name + "_tag_identifier")
torch.save(word_identifier, args.model_name + "_word_identifier")
torch.save(char_identifier, args.model_name + "_char_identifier")

wfeat_dim = word_embedder.wemb_dim
cfeat_dim = 1

batch_size = 1
wlstm_layers = 1

clstm_layers = 1

n_epochs = args.n_epochs
wlstm_dim = args.wlstm_dim
wemb_dim = args.fintetuned_embedding_dim

clstm_dim = args.clstm_dim
cemb_dim = args.fintetuned_cembedding_dim

crf_ip_dim = args.crf_ip_dim

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

def run_epoch(epoch_id):
    batch_loss = A.Variable(torch.zeros(1))
    epoch_loss = 0.0
    np.random.shuffle(sample_ids)  # Uncomment this eventually!
    for order_id, sid in enumerate(sample_ids):
        train_instance = train_instances[sid]
        epoch_loss = process_sample(batch_loss, epoch_loss, order_id, train_instance)
        batch_loss = A.Variable(torch.zeros(1))
    print("\nAverage loss for epoch %d = %f" % (epoch + 1, epoch_loss / len(sample_ids)))
    torch.save(neural_crf.state_dict(), args.model_name + "." + str(epoch + 1))
    print("Saved model after epoch " + str(epoch_id))

# @do_profile(follow=[])
def process_sample(batch_loss, epoch_loss, order_id, train_instance):
    emissions = neural_crf(wids=train_instance.wids, wfeats=train_instance.wfeats, cids=train_instance.cids,
               cfeats=train_instance.cfeats)
    crf_loss = neural_crf.infer_forward(emissions, label_ids=train_instance.tags, label_identifier=tag_identifier)
    batch_loss += crf_loss
    epoch_loss += crf_loss.data[0]
    # print("Processed %d sentences\r"%(order_id)),
    print("Processed %d sentences" % (order_id))
    if (order_id + 1) % batch_size == 0 or order_id == len(sample_ids) - 1:
        neural_crf.zero_grad()
        batch_loss.backward()
        optimizer.step()
        # try:
        #     print("\nTotal loss for batch within epoch %d = %f" % (epoch+1, batch_loss.data[0]))
        # except:
        #     print("\nCouldn't print loss!")
        batch_loss = A.Variable(torch.zeros(1))
    return epoch_loss


for epoch in range(n_epochs):
    run_epoch(epoch)


print("DONE!")

# NOTE THAT A USEFUL FOLLOW UP TO THIS IS TO IMPLEMENT A STRUCTURED PERCEPTRON LOSS! THIS IS THE DIFFERENCE BETWEEN THE
# VITERBI PATH SCORE AND THE GOLD STANDARD PATH SCORE