use_cuda = False
rnd_seed = 1
import torch.autograd as A, torch.nn as N, torch.nn.functional as F
import numpy as np

if not use_cuda:
    import torch
else:
    import torch.cuda as torch # This must be done after importing all other modules within torch since torch.cuda does
    # NOT have overridden versions of autograd, nn etc.

torch.manual_seed(rnd_seed)

EPSILON = A.Variable(torch.ones(1)) * 1e-8


class Attentional_Seq_Labeler(N.Module):

    def __init__(self, wvocab_size, wlstm_layers, wlstm_dim, wfeat_dim, wemb_dim, cvocab_size, clstm_layers, clstm_dim,
                 cfeat_dim, cemb_dim, n_tags):
        super(Attentional_Seq_Labeler, self).__init__()
        self.wlstm_layers = wlstm_layers
        self.clstm_layers = clstm_layers
        self.wlstm_dim = wlstm_dim
        self.clstm_dim = clstm_dim
        self.wemb_dim = wemb_dim
        self.cemb_dim = cemb_dim
        self.clstm_ip_dim = cfeat_dim + cemb_dim
        self.wlstm_ip_dim = wfeat_dim + wemb_dim + 2*self.clstm_dim
        self.wvocab_size = wvocab_size
        self.cvocab_size = cvocab_size
        self.n_tags = n_tags
        self.c_embedder = N.Embedding(num_embeddings=self.cvocab_size, embedding_dim=self.cemb_dim, padding_idx=0, max_norm=1.0
                                      # ,sparse=True
                                      )
        self.w_embedder = N.Embedding(num_embeddings=self.wvocab_size, embedding_dim=self.wemb_dim, padding_idx=0, max_norm=1.0
                                      # ,sparse=True
                                      )
        # sparse determines whether sparse updates should be applied to the embedding matrix
        self.clstm = N.LSTM(input_size=self.clstm_ip_dim, hidden_size=self.clstm_dim, num_layers=self.clstm_layers, batch_first=True,
                            bidirectional=True)
        self.wlstm = N.LSTM(input_size=self.wlstm_ip_dim, hidden_size=self.wlstm_dim, num_layers=self.wlstm_layers, batch_first=True,
                            bidirectional=True)
        self.attention = N.Linear(in_features=2*self.wlstm_dim, out_features=1)
        self.classifier = N.Linear(in_features=4 * self.wlstm_dim, out_features=self.n_tags)


    def make_hidden_context_states(self, num_layers, batch_size, hidden_dim, bidir=True):
        init_hiddens = A.Variable(torch.zeros(num_layers * (2 if bidir else 1), batch_size, hidden_dim))
        init_contexts = A.Variable(torch.zeros(num_layers * (2 if bidir else 1), batch_size, hidden_dim))
        return (init_hiddens, init_contexts)

    def pad(self, ip_list, tot_len):
        ip_size = list(ip_list.size())
        llen = ip_size[1]
        if(llen==tot_len):
            return ip_list
        ip_size[1] = tot_len-llen
        if(type(ip_list.data)==torch.LongTensor):
            catvec = A.Variable(torch.from_numpy(np.zeros(ip_size, dtype=np.int64)))
        else:
            catvec = A.Variable(torch.from_numpy(np.zeros(ip_size, dtype=np.float32)))
        retvec = torch.cat([ip_list, catvec], 1)
        return retvec

    def get_packed_char_seqs(self, cids, cfeats):
        lens = [list(cid.size())[-1] for cid in cids]
        max_len = max(lens)
        ips = zip([self.pad(cids[i], max_len) for i in range(len(cids))],
                  [self.pad(cfeats[i], max_len) for i in range(len(cfeats))],
                  lens,
                  range(len(cids))
                  )
        sorted_ips = sorted(ips, reverse=True, key=lambda x: x[2])
        ips = torch.cat([ip[0] for ip in sorted_ips], 0)
        c_embeddings = self.c_embedder(ips)
        cfeats = torch.cat([ip[1] for ip in sorted_ips], 0)
        clstm_ips = torch.cat([c_embeddings, cfeats], 2)
        lens = [ip[2] for ip in sorted_ips]
        orig_indices = [ip[3] for ip in sorted_ips]
        packed_seq = N.utils.rnn.pack_padded_sequence(clstm_ips, lens, batch_first=True)
        init_hiddens, init_contexts = self.make_hidden_context_states(self.clstm_layers, len(cids), self.clstm_dim)
        return packed_seq, lens, orig_indices, init_hiddens, init_contexts

    def unpack_char_seqs(self, packed_ips, orig_indices):
        unpacked_seq, lens = N.utils.rnn.pad_packed_sequence(packed_ips, batch_first=True)
        num_entries = list(unpacked_seq.size())[0]
        vars = [None for i in range(num_entries)]
        for i in range(num_entries):
            vars[orig_indices[i]] = unpacked_seq[i][lens[i]-1].unsqueeze(0)
        return torch.cat(vars, 0).unsqueeze(0)

    # @do_profile(follow=[])  # not following any functions recursively for now
    def forward(self, wids, wfeats, cids, cfeats):
        """
        Applies the forward pass which can be used while training or testing. It returns the CRF emission scores for
        each word in the sentence. You can then use the get_train_loss method to get the training crf loss associated
        with the gold label sequence for a given input sequence; or the annotate method to get the label sequence
        associated with it.
        :param wids: Ids fo the words of the sentence. It has dim (1, num_words)
        :param wfeats: Features for each word in the sentence. It has dim (1, num_words, wfeat_dim)
        :param cids: Ids for each char of each word. It is a list of tensors of dim (1, wchar_len)
        :param cfeats: Features for each char of each word. It is a list of tensors of dim (1, wchar_len, cfeat_dim)
        :return: returns the crf emission matrix
        """
        wembs = self.w_embedder(wids) # This should now have dim (1, num_words, wemb_dim)
        if wfeats is not None:
            wembs = torch.cat([wembs, wfeats], 2) # wembs should now have dim (1, num_words, wemb_dim+wfeat_dim)
        packed_ips, lens, orig_indices, init_hiddens, init_contexts = self.get_packed_char_seqs(cids, cfeats)
        cembs, (h, c) = self.clstm(packed_ips, (init_hiddens, init_contexts))
        cembs = self.unpack_char_seqs(cembs, orig_indices) # cembs should now have dim (1, num_words, clstm_dim*2)
        wembs = torch.cat([cembs, wembs], 2) # wembs should now have dim (1, num_words, wemb_dim+wfeat_dim+2*clstm_hid)
        token_reps, hn = self.wlstm(wembs, self.make_hidden_context_states(num_layers=self.wlstm_layers,
                                                                            batch_size=1,hidden_dim=self.wlstm_dim))
        # token_reps should now have dim (1, num_words, 2*wlstm_hid)
        weights = F.softmax(self.attention(F.tanh(token_reps)), 1) # should have size(1, num_words, 1)
        att_vec = torch.sum(token_reps*weights,dim=1) # should have dim (1, 2*wlstm_hid)
        att_vecs = torch.cat([att_vec.unsqueeze(0) for i in range(token_reps.data.size()[1])], 1) # should have dim (1, num_words, 2*wlstm_hid)
        token_reps = torch.cat([att_vecs, token_reps], 2) # should have dim (1, num_words, 4*wlstm_hid)
        return self.classifier(token_reps)

    def get_train_loss(self, wids, wfeats, cids, cfeats, label_ids):
        """
        :param wids: Ids fo the words of the sentence. It has dim (1, num_words)
        :param wfeats: Features for each word in the sentence. It has dim (1, num_words, wfeat_dim)
        :param cids: Ids for each char of each word. It is a list of tensors of dim (1, wchar_len)
        :param cfeats: Features for each char of each word. It is a list of tensors of dim (1, wchar_len, cfeat_dim)
        :param label_ids: The gold ids. It should be a torch Variable with size (1, num_words, 1)
        :param label_identifier:
        :return:
        """
        emissions = self.forward(wids, wfeats, cids, cfeats)
        # log_softmax = F.log_softmax(emissions, 2)
        log_softmax = torch.log(F.softmax(emissions, 2) + EPSILON)
        gathered_vals = torch.gather(log_softmax, 2, label_ids)
        gathered_sum = torch.mean(gathered_vals)
        return -gathered_sum


    def annotate(self, wids, wfeats, cids, cfeats):
        """
        Annotates a single instance
        Returns the best scoring label sequence out of all possible label sequences
        :param crf_emissions: crf_emissions should be the torch variable corresponding to the emission scores for each
        word in the sentence. It should have shape (1, num_words, n_tags)
        :return: the crf loss corresponding to the log probabilitiy of the proposed label sequence for the given input
        """
        # assert False, "Need to fix the 1-based indexing for the labels used with crf emissions and need to fix the " \
        #               "safety checks for START and END in the inner loop"
        emissions = self.forward(wids, wfeats, cids, cfeats)
        class_probabs = F.softmax(emissions, 2) # should have dimension (1, num_words, label_set_size)
        chosen_probabs, chosen_labels = torch.max(class_probabs, 2) # chooses the maximum probability class for each word
        # should have size (1, num_words)
        sample_labels = []
        sample_labels = []
        for i in range(class_probabs.size()[1]):
            sample_labels.append(int(chosen_labels[0][i][0].data+1)) # since the label identifier is uses
            # 1 as its base index
        return sample_labels