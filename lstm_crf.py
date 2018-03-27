use_cuda = False
rnd_seed = 1
import torch.autograd as A, torch.nn as N, torch.nn.functional as F
import numpy as np

try:
    from line_profiler import LineProfiler

    def do_profile(follow=[]):
        def inner(func):
            def profiled_func(*args, **kwargs):
                try:
                    profiler = LineProfiler()
                    profiler.add_function(func)
                    for f in follow:
                        profiler.add_function(f)
                    profiler.enable_by_count()
                    return func(*args, **kwargs)
                finally:
                    profiler.print_stats()
            return profiled_func
        return inner

except ImportError:
    def do_profile(follow=[]):
        "Helpful if you accidentally leave in production!"
        def inner(func):
            def nothing(*args, **kwargs):
                return func(*args, **kwargs)
            return nothing
        return inner


if not use_cuda:
    import torch
else:
    import torch.cuda as torch # This must be done after importing all other modules within torch since torch.cuda does
    # NOT have overridden versions of autograd, nn etc.

torch.manual_seed(rnd_seed)

# In this script, a neural CRF is implemented which works using SGD and only utilizes word embeddings and word LSTMs
# (i.e. no character lstms) char lstms aren't hard to implement. They just make batching harder.

# As an advanced exercise, one could make a new PyTorch layer for CRFs that can be added onto any model the produces a
# representation for each item in a sequence and expects contiguous label spans corresponding to these items.

class Neural_CRF(N.Module):

    def __init__(self, wvocab_size, wlstm_layers, wlstm_dim, wfeat_dim, wemb_dim, cvocab_size, clstm_layers, clstm_dim, cfeat_dim,
                 cemb_dim, crf_ip_dim, n_tags):
        super(Neural_CRF, self).__init__()
        clstm_ip_dim = cfeat_dim + cemb_dim
        wlstm_ip_dim = wfeat_dim + wemb_dim + 2*clstm_dim
        self.n_tags_with_start_end = n_tags + 2
        self.wvocab_size = wvocab_size
        self.cvocab_size = cvocab_size
        self.NINF = -10000
        self.START=0
        self.END = n_tags+1
        self.c_embedder = N.Embedding(num_embeddings=cvocab_size, embedding_dim=cemb_dim, padding_idx=0, max_norm=1.0
                                      # ,sparse=True
                                      )
        self.w_embedder = N.Embedding(num_embeddings=wvocab_size, embedding_dim=wemb_dim, padding_idx=0, max_norm=1.0
                                      # ,sparse=True
                                      )
        # sparse determines whether sparse updates should be applied to the embedding matrix
        self.clstm = N.LSTM(input_size=clstm_ip_dim, hidden_size=clstm_dim, num_layers=clstm_layers, batch_first=True,
                            bidirectional=True)
        self.wlstm = N.LSTM(input_size=wlstm_ip_dim, hidden_size=wlstm_dim, num_layers=wlstm_layers, batch_first=True,
                            bidirectional=True)
        self.pre_emitter = N.Linear(in_features=2*wlstm_dim, out_features=crf_ip_dim)
        self.emitter = N.Linear(in_features=crf_ip_dim, out_features=n_tags)
        self.transition_scores = N.Linear(in_features=n_tags+2, out_features=n_tags+2, bias=False) # parameter matrix for
        # the CRF. Each parameter determines the compatibility between the corresponding two tags occurring next to each
        # other
        for tag_id in range(self.n_tags_with_start_end):
            self.transition_scores.weight.data[tag_id][self.START] = self.NINF # Transition into the start tag should be
            # impossible
            self.transition_scores.weight.data[self.END][tag_id] = self.NINF # Transitioning out of the end tag should be
            # impossible

    def log_sum_exp(self, log_sum_exp_elements):
        """
        Finds the log sum exp of the provided elements in a numerically stable fashion. Refer to
        https://hips.seas.harvard.edu/blog/2013/01/09/computing-log-sum-exp/ for the justification behind this method
        :param log_sum_exp_elements: The torch variable of elements that we want to find the log sum exp of. It should
                                     have dim (num_elements,)
        :return: The torch float tensor corresponding to the single float value of the log sum exp result
        """
        max_elt = torch.max(log_sum_exp_elements)
        max_vec = max_elt.expand(log_sum_exp_elements.size()[-1])
        return max_elt + torch.log(torch.sum(torch.exp(log_sum_exp_elements-max_vec)))

    @do_profile(follow=[])  # not following any functions recursively for now
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

        # now assembling the character embedding
        cembs = []
        for i in range(len(cids)):
            clstm_ip = torch.cat([self.c_embedder(cids[i]), cfeats[i]], 2) # this should have dim (1, w_char_len,
            # cemb_dim+cfeat_dim)
            cemb, hn = self.clstm(clstm_ip) # cemb should have dim (1, w_char_len, clstm_dim*2)
            # cemb = cemb.view(cemb.size(1), cemb.size(0), cemb.size(2))[:,-1,:].unsqueeze(0)
            cemb = cemb[:, -1, :].unsqueeze(0) # cemb should now have dim (1, 1, clstm_dim*2)
            cembs.append(cemb)

        cembs = torch.cat(cembs, 1) # cembs should now have dim (1, num_words, clstm_dim*2)
        wembs = torch.cat([cembs, wembs], 2) # wembs should now have dim (1, num_words, wemb_dim+wfeat_dim+2*clstm_hid)
        token_reps, hn = self.wlstm(wembs)
        crf_emissions = self.emitter(
                            F.tanh(
                                self.pre_emitter(token_reps.view(-1, token_reps.size()[-1]))
                                )
                            ).unsqueeze(0) # This should have dimension
        # The reshaping via view is necessary since a linear layer can't handle 3D tensors
        # (1, num_words, n_tags). Note that n_tags here also includes the start of end symbols
        return crf_emissions

    # @do_profile(follow=[log_sum_exp]) # not following any functions recursively for now
    def infer_forward(self, crf_emissions, label_ids, label_identifier):
        """
        Returns the log likelihood associated with the gold label sequence. Note that this requires the execution of the
        forward algorithm which incurs a performance penalty linear in the length of the sentence and quadratic in the
        number of tags. YOU MUST NEGATE THE VALUE RETURNED TO OBTAIN THE LOSS VALUE!
        :param label_ids: ids corresponding to the labels of the ner tags for each word in the sentence. This should be
        a list of ids. The ids are assumed to be indexed with base 1
        :param crf_emissions: crf_emissions should be the torch variable corresponding to the emission scores for each
        word in the sentence. It should have shape (1, num_words, n_tags)
        :return: the crf loss corresponding to the log probabilitiy of the proposed label sequence for the given input
        """
        if len(label_ids) == crf_emissions.size(1)+1:
            label_ids.append(label_identifier.vocab_size+1)
        assert len(label_ids) == crf_emissions.size(1)+2, "Number of labels should be the number of words in " \
                                                        "the sentence + 2"

        # first we find the numerator term
        score = A.Variable(torch.zeros(1))
        for i in range(len(label_ids)):
            if i>0 and i<len(label_ids)-1:
                score += crf_emissions[0,i-1,label_ids[i]-1] # label ids are assumed to be 1 indexed in the input
            if i+1 < len(label_ids):
                score += self.transition_scores.weight[label_ids[i],label_ids[i+1]]

        # Now you need to use the forward algorithm to get the normalization factor
        prev_log_alphas = A.Variable(torch.FloatTensor([self.NINF for i in range(self.n_tags_with_start_end)]))
        prev_log_alphas[0] = A.Variable(torch.zeros(1)) # Initially, the starting symbol should have the highest score
        for i in range(len(label_ids)-2): # starting with the first word
            curr_log_alphas = A.Variable(torch.zeros(self.n_tags_with_start_end))
            for curr_tag_id in range(self.n_tags_with_start_end):
                log_sum_exp_elements = A.Variable(torch.zeros(self.n_tags_with_start_end))
                for prev_tag_id in range(self.n_tags_with_start_end):
                    candidate_score = prev_log_alphas[prev_tag_id] + \
                                      self.transition_scores.weight[prev_tag_id, curr_tag_id]
                    # do not use self.transition_scores.weight[prev_tag_id][curr_tag_id] since that involves the
                    # creation of an intermediate tensor. Since only one such intermediate tensor can be created at a
                    # time, it will cause errors
                    if (curr_tag_id!=self.END and curr_tag_id!=self.START):
                        candidate_score += crf_emissions[0, i, curr_tag_id - 1]
                        # The start and end symbols cannot be emitted by any word
                    else:
                        candidate_score += A.Variable(torch.FloatTensor([self.NINF]))[0]
                    log_sum_exp_elements[prev_tag_id] = candidate_score
                curr_log_alphas[curr_tag_id] = self.log_sum_exp(log_sum_exp_elements)
            prev_log_alphas = curr_log_alphas
        log_sum_exp_elements = A.Variable(torch.zeros(self.n_tags_with_start_end))
        for prev_tag_id in range(self.n_tags_with_start_end):
            log_sum_exp_elements[prev_tag_id] = prev_log_alphas[prev_tag_id] + \
                                                self.transition_scores.weight[prev_tag_id, self.END]
            # no emission potential added since the end symbol doesn't need to be emitted!

        score -= self.log_sum_exp(log_sum_exp_elements) # The score should now be the log likelihood of the
        # correct label
        return -score

    def annotate(self, crf_emissions):
        """
        Returns the best scoring label sequence out of all possible label sequences
        :param crf_emissions: crf_emissions should be the torch variable corresponding to the emission scores for each
        word in the sentence. It should have shape (1, num_words, n_tags)
        :return: the crf loss corresponding to the log probabilitiy of the proposed label sequence for the given input
        """
        # assert False, "Need to fix the 1-based indexing for the labels used with crf emissions and need to fix the " \
        #               "safety checks for START and END in the inner loop"
        sent_len = crf_emissions.size(1)
        # prev_pointers = [[-1 for wpos in range(self.n_tags_with_start_end)] for wpos in range(sent_len)]
        prev_pointers = np.zeros(shape=(sent_len, self.n_tags_with_start_end))
        # Now you need to use the viterbi algorithm to get the normalization factor
        prev_log_alphas = A.Variable(torch.FloatTensor([self.NINF for tid in range(self.n_tags_with_start_end)]))
        prev_log_alphas[0] = A.Variable(torch.zeros(1)) # Initially, the starting symbol should have the highest score
        for wpos in range(sent_len): # starting with the first word
            curr_log_alphas = A.Variable(torch.zeros(self.n_tags_with_start_end))
            for curr_tag_id in range(self.n_tags_with_start_end):
                for prev_tag_id in range(self.n_tags_with_start_end):
                    candidate_score = prev_log_alphas[prev_tag_id] + \
                                      self.transition_scores.weight[prev_tag_id, curr_tag_id]
                    if curr_tag_id!=self.START and curr_tag_id!=self.END:
                        candidate_score += crf_emissions[0, wpos, curr_tag_id-1]
                    else:
                        candidate_score += A.Variable(torch.FloatTensor([self.NINF]))[0]
                    # print("Curr log alpha val for curr_tag %d = %f"%(curr_tag_id, curr_log_alphas.en-fr_data[curr_tag_id]))
                    # print("Candidate score = %f" % (candidate_score.en-fr_data[0]))
                    # print("Comparison result is ", curr_log_alphas.en-fr_data[curr_tag_id] < candidate_score.en-fr_data[0])
                    if prev_tag_id == 0:
                        curr_log_alphas[curr_tag_id] = candidate_score
                        prev_pointers[wpos][curr_tag_id] = prev_tag_id
                    # elif curr_log_alphas[curr_tag_id] < candidate_score: # this comparison is incorrect for some reason
                    elif curr_log_alphas.data[curr_tag_id] < candidate_score.data[0]:
                        curr_log_alphas[curr_tag_id] = torch.max(curr_log_alphas[curr_tag_id], candidate_score)
                        prev_pointers[wpos][curr_tag_id] = prev_tag_id
            prev_log_alphas = curr_log_alphas

        final_tag_score, final_tag_id = torch.max(curr_log_alphas, 0)
        # no emission potential added since the end symbol doesn't need to be emitted!
        labels = [self.END, final_tag_id.data[0]]
        for i in range(sent_len-1,-1,-1):
            labels.append(int(prev_pointers[i][labels[-1]]))

        return labels[::-1], final_tag_score