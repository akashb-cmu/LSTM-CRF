import numpy as np
from codecs import open
import torch.autograd as A

class Word_Embedder():

    def __init__(self, emb_file):
        self.wembs = {}
        self.wemb_dim = 0
        with open(emb_file, 'r', encoding='utf-8') as ipf:
            line_no = 0
            for line in ipf:
                line = line.strip(' \t\r\n')
                if len(line)<1:
                    continue
                line_no+=1
                line = line.split()
                self.wemb_dim = len(line) - 1
                self.wembs[line[0]] = np.array([float(dim) for dim in line[1:]], dtype=np.float32)
                print("Read %d lines\r"%(line_no+1)),
        print("Done reading all word embeddings")

    def get_emb(self, word):
        return self.wembs.get(word, np.zeros(shape=(self.wemb_dim), dtype=np.float32))

class Identifer():

    def __init__(self):
        self.word2wid = {}
        self.wid2word = {}
        self.vocab_size = 0

    def get_id(self, word, is_train):
        if self.word2wid.has_key(word):
            return self.word2wid[word]
        else:
            if is_train:
                self.word2wid[word] = len(self.word2wid) + 1
                self.vocab_size += 1
                self.wid2word[self.word2wid[word]] = word
                return self.word2wid[word]
            else:
                return 0

    def get_word(self, id):
        return self.wid2word.get(id, "$<UNK>$")

class Instance():

    def __init__(self, instance_lines, word_embedder, word_identifier, char_identifier, tag_identifier,
                 is_train, use_cuda=False):
        self.instance_lines = instance_lines
        self.wids = [] # should be a torch variable of dim (1, num_words)
        self.wfeats = [] # should be a torch variable of dim (1, num_words, wfeat_size)
        self.cids = [] # should be a list of tensors of dim (1, wchar_len)
        self.cfeats = [] # should be a list of tensors of dim (1, wchar_len, cfeat_dim)
        self.tags = [0] # should just be a list of ids, where tags are indexed with base 1
        if use_cuda:
            import torch.cuda as torch
        else:
            import torch
        for line in instance_lines:
            line = line.strip(' \t\r\n')
            if len(line) == 0:
                continue
            line = line.split()
            word = line[0]
            tag = line[-1]
            wid = word_identifier.get_id(word, is_train)
            wemb = word_embedder.get_emb(word)
            tid = tag_identifier.get_id(tag, is_train)
            self.wids.append(wid)
            self.tags.append(tid)
            self.wfeats.append(wemb)
            word_cids = []
            word_cfeats = []
            for char in word:
                cid = char_identifier.get_id(char, is_train)
                word_cids.append(cid)
                word_cfeats.append([int(char.isupper())])
            self.cids.append(A.Variable(torch.LongTensor(word_cids)).unsqueeze(0))
            self.cfeats.append(A.Variable(torch.FloatTensor(word_cfeats)).unsqueeze(0))
        self.wids = A.Variable(torch.LongTensor(self.wids)).unsqueeze(0)
        self.wfeats = A.Variable(torch.from_numpy(np.array(self.wfeats))).unsqueeze(0)

    def get_output_conll(self, pred_labels_ids, tag_identifier):
        pred_output_labels = pred_labels_ids[1:-1]
        output_lines = []
        for iid, instance_line in enumerate(self.instance_lines):
            instance_line = instance_line.strip(' \t\r\n')
            if len(instance_line)==0:
                continue
            output_line = instance_line.split()
            output_line[-1] = tag_identifier.get_word(pred_output_labels[iid])
            output_lines.append(output_line)
        return "\n".join([" ".join(line) for line in output_lines])


def get_ner_samples(train_file, dev_file, test_file, emb_file, word_embedder=None, word_identifier=None,
                    tag_identifier=None, char_identifier=None, use_pretrained=False, use_cuda=False):
    word_embedder = Word_Embedder(emb_file) if word_embedder is None else word_embedder
    word_identifier = Identifer() if word_identifier is None else word_identifier
    tag_identifier = Identifer() if tag_identifier is None else tag_identifier
    char_identifier = Identifer() if char_identifier is None else char_identifier
    train_instances = get_instances(train_file, word_embedder, word_identifier, tag_identifier, char_identifier, not use_pretrained, use_cuda)
    dev_instances = get_instances(dev_file, word_embedder, word_identifier, tag_identifier, char_identifier, False, use_cuda)
    test_instances = get_instances(test_file, word_embedder, word_identifier, tag_identifier, char_identifier, False, use_cuda)
    return train_instances, dev_instances, test_instances, word_embedder, word_identifier, tag_identifier, char_identifier

def get_instances(file_path, word_embedder, word_identifier, tag_identifier, char_identifier, is_train, use_cuda=False):
    if not use_cuda:
        import torch
    else:
        import torch.cuda as torch # This must be done after importing all other modules within torch since torch.cuda does
        # NOT have overridden versions of autograd, nn etc.
    samples = []
    print("Caution: At this point, unkification is not done while processing the training set. This can lead to "
          "issues if the test vocabulary deviates from the train set in a non-negligible way. In order to ensure the"
          " model is robust to the occurrence of new words, unkification is a must!")
    print("Need to correct the tagging scheme to BIO!")
    with open(file_path, 'r', encoding='utf-8') as ipf:
        instances = ipf.read().split('\n\n')
        for iid, instance in enumerate(instances):
            instance = instance.strip(" \t\r\n")
            if len(instance)>0:
                samples.append(Instance(instance.split('\n'), word_embedder, word_identifier, char_identifier,
                                        tag_identifier, is_train, use_cuda))
                print("Processed %d samples\r"%(iid+1)),
    print("")
    return samples



def annotate_corpus(output_file, test_instances, neural_crf, tag_identifier):
    with open(output_file, 'w+', encoding='utf-8') as opf:
        for test_instance in test_instances:
            pred_labels, final_tag_score = neural_crf.annotate(neural_crf(wids=test_instance.wids,
                                                                          wfeats=test_instance.wfeats,
                                                                          cids=test_instance.cids,
                                                                          cfeats=test_instance.cfeats))
            # compare label_ids=tags[sid] and pred_labels
            opf.write(test_instance.get_output_conll(pred_labels, tag_identifier) + "\n\n")