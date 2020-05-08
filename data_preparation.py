import jsonlines
import tensorflow as tf

val = {}
files = ['tvqa_train.jsonl', 'tvqa_val.jsonl']

with tf.io.gfile.GFile("tvqa_train.tsv", "w") as outfile:
      with jsonlines.open('tvqa_train.jsonl') as f:

            for line in f.iter():
                  answer = "a"+str(line['answer_idx'])
                  ques = line['q'].strip()[:-1]
                  if(ques[-1] == "?"):
                        ques = ques[:-1]
                  outfile.write("%s\t%s\n" % (ques, line[answer]))

      with jsonlines.open('tvqa_val.jsonl') as f1:

            for line in f1.iter():
                  answer = "a"+str(line['answer_idx'])
                  ques = line['q'].strip()[:-1]
                  if(ques[-1] == "?"):
                        ques = ques[:-1]
                  outfile.write("%s\t%s\n" % (ques, line[answer]))
