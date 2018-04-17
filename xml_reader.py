from bs4 import BeautifulSoup
import codecs
import os
import argparse
import numpy as np
import math

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--ip_dir_list", nargs='+', help="SPecifiy a space separated list of directories in which"
                    " the WSD data xml files are present")
parser.add_argument("-o", "--output_file", type=str,
                default=None)

args = parser.parse_args()
ip_dir_list = args.ip_dir_list

def process_word(lines, samples, concat_levels, tokenize_levels):
    pos = word.get("pos", "_")
    sense = "B-"
    if(word.has_attr("sense")):
        sense += word["sense"].split('/')[-1]
    else:
        sense += "0"
    token = word["text"]
    brk = word["break_level"]
    line = [token, pos, sense]
    if(brk in concat_levels):
        if(len(lines)>0):
            lines[-1][0] += token # assuming that the sense and pos are same for hte two parts of the same word
            if lines[-1][1] == "_":
                lines[-1][1] = pos
            if lines[-1][-1] == "B-O":
                lines[-1][-1] = sense
        else:
            lines.append(line)
    elif(brk in tokenize_levels):
        lines.append(line)
    else:
        samples.append("\n".join(["\t".join(sline) for sline in lines]))
        lines[:] = [] # clearing the sample's lines to accommodate the next sample
        lines.append(line)
    return id

sentence_break_levels = set(["SENTENCE_BREAK", "PARAGRAPH_BREAK"])
concat_levels = set(["NO_BREAK"])
tokenize_levels = set(["SPACE_BREAK", "LINE_BREAK"])
train_samples = []
dev_samples = []
test_samples = []

files_processed = 0
num_words = 0
for dir in ip_dir_list:
    for (path, dirs, files) in os.walk(dir, topdown=True):
        print("In dir " + path)
        for file in files:
            file_samples = []
            file_train_samples = []
            file_test_samples = []
            file_dev_samples = []
            if(file.endswith(".xml")):
                files_processed += 1
                source_xml = codecs.open(os.path.join(path, file), 'r', encoding='utf8').read()
                xml_reader = BeautifulSoup(source_xml, features="xml")
                words = xml_reader.find_all("word")
                # print("num words = ", len(words))
                num_words += len(words)
                lines = []
                for word in words:
                    id = process_word(lines, file_samples, concat_levels, tokenize_levels)
                if len(lines)>0:
                    file_samples.append("\n".join(["\t".join(sline) for sline in lines]))
                num_samples = int(min(
                                         [
                                             math.ceil(float(len(file_samples))*0.2),
                                             max(len(file_samples)-1, 0)
                                         ]
                                    )
                                )
                dev_test_indices = []
                if num_samples>0:
                    dev_test_indices = np.random.choice(range(len(file_samples)), num_samples)
                if(len(dev_test_indices)>0):
                    for i in range(len(dev_test_indices)):
                        if i%2==0:
                            file_dev_samples.append(file_samples[dev_test_indices[i]])
                        else:
                            file_test_samples.append(file_samples[dev_test_indices[i]])
                dev_test_indices = set(dev_test_indices)
                for i in range(len(file_samples)):
                    if(i not in dev_test_indices):
                        file_train_samples.append(file_samples[i])
                train_samples.extend(file_train_samples)
                dev_samples.extend(file_dev_samples)
                test_samples.extend(file_test_samples)

with codecs.open(args.output_file + "_train.conll", 'w', encoding='utf8') as opf:
    opf.write("\n\n".join(train_samples))

with codecs.open(args.output_file + "_dev.conll", 'w', encoding='utf8') as opf:
    opf.write("\n\n".join(dev_samples))

with codecs.open(args.output_file + "_test.conll", 'w', encoding='utf8') as opf:
    opf.write("\n\n".join(test_samples))

print("Files processed: %d"%(files_processed))
print("words processed: %d"%(num_words))
print("Tot number of train samples found: %d"%(len(train_samples)))
print("Tot number of dev samples found: %d"%(len(dev_samples)))
print("Tot number of test samples found: %d"%(len(test_samples)))