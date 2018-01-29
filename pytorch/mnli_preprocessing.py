import os
import json
import codecs
import argparse
from collections import defaultdict

"""
Transforms MNLI data into lines of text files
    (data format required for ARAE model).
Gets rid of repeated premise sentences.
"""

def transform_data(in_path):
    print("Loading", in_path)

    genre2premises = defaultdict(list)
    genre2hypotheses = defaultdict(list)

    last_premise = None
    with codecs.open(in_path, encoding='utf-8') as f:
        for line in f:
            loaded_example = json.loads(line)
            genre = loaded_example['genre']

            # load premise
            raw_premise = loaded_example['sentence1_binary_parse'].split(" ")
            premise_words = []
            # loop through words of premise binary parse
            for word in raw_premise:
                # don't add parse brackets
                if word != "(" and word != ")":
                    premise_words.append(word)
            premise = " ".join(premise_words)

            # load hypothesis
            raw_hypothesis = \
                loaded_example['sentence2_binary_parse'].split(" ")
            hypothesis_words = []
            for word in raw_hypothesis:
                if word != "(" and word != ")":
                    hypothesis_words.append(word)
            hypothesis = " ".join(hypothesis_words)

            # make sure to not repeat premises
            if premise != last_premise:
                genre2premises[genre].append(premise)
            genre2hypotheses[genre].append(hypothesis)

            last_premise = premise

    return genre2premises, genre2hypotheses


def write_sentences(out_path, write_path, genre2premises, genre2hypotheses, append=False):
    mode = 'a' if append else 'w'
    for genre, premises in genre2premises.items():
        print("Writing to {}/{}\n".format(out_path, write_path).format(genre))
        
        # make genre out-path directory if it doesn't exist
        if not os.path.exists(out_path.format(genre)):
            os.makedirs(out_path.format(genre))
            print("Creating directory "+out_path.format(genre))

        with codecs.open(os.path.join(out_path.format(genre), write_path), mode, encoding='utf-8') as f:
            for p in premises:
                f.write(p)
                f.write("\n")
    # Ignore genre2hypotheses data. The premises are drawn from genre corpora, 
    # the hypotheses are produced by cloud workers.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, default="../Data/multinli_1.0",
                        help='path to mnli data')
    parser.add_argument('--out_path', type=str, default="../Data/multinli_lm_{}",
                        help='path to write mnli language modeling data to')
    args = parser.parse_args()

    # process and write test.txt and train.txt files
    genre2premises, genre2hypotheses = \
        transform_data(os.path.join(args.in_path, "multinli_1.0_train.jsonl"))
    write_sentences(out_path=args.out_path,
                    write_path="train.txt",
                    genre2premises=genre2premises, 
                    genre2hypotheses=genre2hypotheses)
    genre2premises, genre2hypotheses = \
        transform_data(os.path.join(args.in_path, "multinli_1.0_dev_matched.jsonl"))
    write_sentences(out_path=args.out_path,
                    write_path="test.txt",
                    genre2premises=genre2premises,
                    genre2hypotheses=genre2hypotheses)

    # genre2premises, genre2hypotheses = \
    #     transform_data(os.path.join(args.in_path, "multinli_1.0_dev_matched.jsonl"))
    # write_sentences(out_path=args.out_path,
    #                 write_path="train.txt",
    #                 genre2premises=premises,
    #                 genre2hypotheses=hypotheses,
    #                 append=True)
