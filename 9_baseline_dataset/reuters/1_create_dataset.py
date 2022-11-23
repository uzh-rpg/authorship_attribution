""" Short Summary
This script takes the Reuters-50-50 dataset from
      https://archive.ics.uci.edu/ml/datasets/Reuter_50_50
and processes it into the standard form we use in this project. Namely, the
following files are generated:
    - content.csv
    - features.csv
    - paper_ids.csv
    - authors.txt
    - author2id.csv
The are generated inside the "dataset_50_50/train" and "dataset_50_50/test"
folder respectively.
"""

import os
import re
import random
import numpy as np

def getFiles(train_test):
    """ Returns a list of all files in the train or test directory.
    train_test: a string that decides whether the train or test data is used.
                it can either be "test" or "train"

    Returns:
    filelist:   a list of all files
    """
    assert(train_test == "train" or train_test == "test")
    authors = []
    filelist = []
    for path, dirnames, filenames in os.walk('Reuters_50_50/' + train_test):
        for filename in filenames:
            filelist.append(os.path.join(path , filename))

    return filelist


def extractData(filelist):
    """ Extracts the author, article id and author-article association based
        based on the filename
    filelist:    list of all filenames

    Returns:
    paper_ids:   list of all article identifier numbers
    author2id:   dictionary: author name -> list of article ids he wrote
    id2author:   dictionary: article id -> author name
    """
    author2id = {}
    id2author = {}
    paper_ids = []
    for f in filelist:
        author = f.split("/")[-2]
        paper_id = re.search(r"^\d+", f.split("/")[-1]).group(0)
        if author not in author2id.keys():
            author2id[author] = [int(paper_id)]
        else:
            author2id[author].append(int(paper_id))
        id2author[paper_id] = author
        paper_ids.append(paper_id)

    random.shuffle(paper_ids)

    for key in author2id.keys():
        author2id[key] = [str(x) for x in sorted(author2id[key])]

    return author2id, id2author, paper_ids


def makeFeatures(authors, paper_ids, id2author):
    """ Create one-hot vector encoding the author for each paper
    authors, paper_ids, id2author as produced by extractData function
    Returns:
    feature_vec: numpy array where each line represents one paper
                 the nth number is 1 (instead of 0) meaning thath
                 this document was writetn by the nth author from the
                 author list.
    """
    feature_vec = np.zeros((len(paper_ids), len(authors)))
    for i in range(len(paper_ids)):
        feature_vec[i, authors.index(id2author[paper_ids[i]])] = 1
    return feature_vec


def getContent(paper_ids, id2author, train_test):
    """ Reads all article files
    Returns:
    content:    a list where each entry corresponds to the contents of
                an article. The order of the list is identical to paper_ids
                and all articles are stripped of newlines.
    """
    content = []
    for paper_id in paper_ids:
        author = id2author[paper_id]
        path = os.path.join("Reuters_50_50",
                            train_test,
                            author,
                            paper_id + "newsML.txt")
        with open(path) as f:
            content.append(f.read().replace("\n"," ").replace("\x0c"," "))
    return content


def writeToFile(authors, features, author2id, paper_ids, content, train_test):
    """ Writes all data into files
    The following files are written (where X is either "test" or "train"
    depending on the value of the argument train_test.
    - dataset_50_50/X/features.csv
    - dataset_50_50/X/authors.txt
    - dataset_50_50/X/paper_ids.csv
    - dataset_50_50/X/author2id.csv
    - dataset_50_50/X/content.csv
    For file specification, see README.md in root directory.
    """
    np.savetxt("dataset_50_50/" + train_test + "/features.csv", features, fmt="%1i")
    np.savetxt("dataset_50_50/" + train_test + "/authors.txt", authors, fmt="%s")
    np.savetxt("dataset_50_50/" + train_test + "/paper_ids.csv", paper_ids, fmt="%s")
    np.savetxt("dataset_50_50/" + train_test + "/author2id.csv",
            np.array(author2id[authors[0]], dtype=str, ndmin=2), fmt="%s", newline='')
    with open("dataset_50_50/" + train_test + "/author2id.csv", "ab") as f:
        f.write(b"\n")
        for i in range(1, len(authors)):
            np.savetxt(f, np.array(author2id[authors[i]], dtype=str, ndmin=2), fmt="%s")
    with open("dataset_50_50/" + train_test + "/content.csv", "w") as f:
        f.write("\n".join(content))
        f.write("\n")


""" Run the program for both train and test folders """
for train_test in ["train", "test"]:
    filelist = sorted(getFiles(train_test))
    author2id, id2author, paper_ids = extractData(filelist)
    authors = list(author2id.keys())
    features = makeFeatures(authors, paper_ids, id2author)
    content = getContent(paper_ids, id2author, train_test)
    writeToFile(authors, features, author2id, paper_ids, content, train_test)
