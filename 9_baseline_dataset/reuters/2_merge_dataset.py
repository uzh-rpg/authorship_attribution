""" Short Summary
This script merges the training and testdata of the Reuters 50/50 dataset. This
is required to try a different train/test split (e.g. 90/10) for comparison with
other works from the literature.
The output is written to dataset-reuters/
"""
import os
import numpy as np

def concatenateFiles(fname):
    """ Concatenate two files (one test, one train) 
    fname:  the name of the files to be concatenated, must exist
            under dataset_50_50/train and dataset_50_50/test folders
    """
    with open("dataset_50_50/train/" + fname) as train,\
         open("dataset_50_50/test/" + fname) as test,\
         open("dataset-reuters/" + fname, "w") as new:
        new.write(train.read() + test.read())

concatenateFiles("content.csv")
concatenateFiles("features.csv")
concatenateFiles("paper_ids.csv")


""" Merge the authors and authors2id files
This  merge process is slightly more involved as the order of the authors
is not stricly guaranteed to be identical. 
1. The authors.txt is used as a key
2. The author2id file is read into a dictionary with the corresponding key.
   This is done for both training and test data, thus both are merged into 
   a single dictionary.
3. Write to file.
"""
with open("dataset_50_50/train/authors.txt") as authors_train,\
     open("dataset_50_50/test/authors.txt") as authors_test,\
     open("dataset_50_50/train/author2id.csv") as author2id_train,\
     open("dataset_50_50/test/author2id.csv") as author2id_test:
    author2id = {}
    for a_train, a2i_train in zip(authors_train.readlines(),
                                  author2id_train.readlines()):
        a_train = a_train.strip()
        a2i_train = a2i_train.strip()
        author2id[a_train] = a2i_train.split(" ")

    for a_test, a2i_test in zip(authors_test.readlines(),
                               author2id_test.readlines()):
        a_test = a_test.strip()
        a2i_test = a2i_test.strip()
        author2id[a_test].extend(a2i_test.split(" "))

    authors = list(author2id.keys())
    np.savetxt("dataset-reuters/authors.txt", authors, fmt="%s")
    np.savetxt("dataset-reuters/author2id.csv", 
            np.array(author2id[authors[0]], dtype=str, ndmin=2), fmt="%s", newline='')
    with open("dataset-reuters/author2id.csv", "ab") as f:
        f.write(b"\n")
        for i in range(1, len(authors)):
            np.savetxt(f, np.array(author2id[authors[i]], dtype=str, ndmin=2), fmt="%s")
