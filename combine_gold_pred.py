import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-tr", "--pred_test_file", type=str,
                    default="/home/akashb/Desktop/Blog projects/LSTM_CRF/annotations/output_resumed_again.conll.3")
parser.add_argument("-te", "--gold_test_file", type=str,
                    default="/home/akashb/Desktop/Blog projects/common-resources/conll_data/eng_data/eng.testb.true.conll")
parser.add_argument("-c", "--combined_file", type=str,
                    default="/home/akashb/Desktop/Blog projects/LSTM_CRF/evaluation/output_resumed_again.conll.3.conll")


args = parser.parse_args()

pred_test_file = args.pred_test_file
gold_test_file = args.gold_test_file
op_file = args.combined_file

op_instance_strs = []

with open(gold_test_file, 'r') as gt:
    with open(pred_test_file, 'r') as pt:
        gold_instances = gt.read().strip(' \t\r\n').split('\n\n')
        pred_instances = pt.read().strip(' \t\r\n').split('\n\n')
        assert len(gold_instances) == len(pred_instances), "Number of instances doesn't match up!"
        for iid, ginstance in enumerate(gold_instances):
            ginstance = ginstance.strip(" \t\r\n")
            if len(ginstance) == 0:
                continue
            pinstance = pred_instances[iid]
            ginstance = ginstance.split('\n')
            pinstance = pinstance.split('\n')
            assert len(ginstance) == len(pinstance), "Number of lines in an instance doesn't match!"
            instance_lines = []
            for lid, gline in enumerate(ginstance):
                pline = pinstance[lid].split()
                gline = gline.split()
                gline.append(pline[-1])
                instance_lines.append(" ".join(gline))
            op_instance_strs.append("\n".join(instance_lines))


with open(op_file, 'w+') as opf:
    opf.write("\n\n".join(op_instance_strs))