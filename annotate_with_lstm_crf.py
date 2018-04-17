import lstm_crf
import argparse
from my_utils import *
import os
# from my_profiler import *
use_cuda = False
rnd_seed = 1

parser = argparse.ArgumentParser()
parser.add_argument("-te", "--test_file", type=str,
                    default="../ner_data/eng_data/eng.testb.true.conll")
parser.add_argument("-e", "--embedding_file", type=str,
                    default="../embeddings/glove.840B.300d.txt")
# parser.add_argument("-e", "--embedding_file", type=str,
#                     default="../embeddings/glove_dummy.txt")
parser.add_argument("-o", "--output_file", type=str,
                    default="./annotations/ner_before_wsd.mod_batch2.5.conll")
parser.add_argument("-p", "--pretrained_model_name", type=str,
                    default="./models/ner_before_wsd_batch2.mod.5")
parser.add_argument("-p2", "--identifer_path_prefix", type=str,
                    default="./models/ner_before_wsd_batch2.mod")
parser.add_argument("--use_cuda", dest="use_cuda", action='store_true')

args = parser.parse_args()
if args.use_cuda:
    use_cuda = True

test_file = args.test_file
embedding_file = args.embedding_file

if not use_cuda:
    import torch
else:
    import torch.cuda as torch # This must be done after importing all other modules within torch since torch.cuda does
    # NOT have overridden versions of autograd, nn etc.
torch.manual_seed(rnd_seed)


tag_identifier = torch.load(args.identifer_path_prefix+"_tag_identifier") if os.path.isfile(args.identifer_path_prefix+"_tag_identifier") else None
char_identifier = torch.load(args.identifer_path_prefix+"_char_identifier") if os.path.isfile(args.identifer_path_prefix+"_char_identifier") else None
word_identifier = torch.load(args.identifer_path_prefix+"_word_identifier") if os.path.isfile(args.identifer_path_prefix+"_word_identifier") else None

_, _, test_instances, word_embedder, word_identifier, tag_identifier, char_identifier = \
    get_ner_samples(None, None, test_file, embedding_file, tag_identifier=tag_identifier,
                    word_identifier=word_identifier, char_identifier=char_identifier, use_cuda=use_cuda,
                    use_pretrained=not (None in [tag_identifier, char_identifier, word_identifier]))

wfeat_dim = word_embedder.wemb_dim
cfeat_dim = 1
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

print("Starting annotation procedure!")

annotate_corpus_with_crf(args.output_file, test_instances, neural_crf, tag_identifier)

print("DONE!")

# NOTE THAT A USEFUL FOLLOW UP TO THIS IS TO IMPLEMENT A STRUCTURED PERCEPTRON LOSS! THIS IS THE DIFFERENCE BETWEEN THE
# VITERBI PATH SCORE AND THE GOLD STANDARD PATH SCORE