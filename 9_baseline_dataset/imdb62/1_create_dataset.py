import os
import numpy as np
import pandas as pd

def writeToFile(authors, features, author2id, paper_ids, content):
    """ Writes all data into files
    The following files are written
    - dataset/features.csv
    - dataset/authors.txt
    - dataset/paper_ids.csv
    - dataset/author2id.csv
    - dataset/content.csv
    For file specification, see README.md in root directory.
    """
    np.savetxt("dataset/" + "features.csv", features, fmt="%1i")
    np.savetxt("dataset/" + "authors.txt", authors, fmt="%s")
    np.savetxt("dataset/" + "paper_ids.csv", paper_ids, fmt="%s")
    np.savetxt("dataset/" + "author2id.csv",
            np.array(author2id[authors[0]], dtype=str, ndmin=2), fmt="%s", newline='')
    with open("dataset/" + "author2id.csv", "ab") as f:
        f.write(b"\n")
        for i in range(1, len(authors)):
            np.savetxt(f, np.array(author2id[authors[i]], dtype=str, ndmin=2), fmt="%s")
    with open("dataset/"+ "content.csv", "w") as f:
        f.write("\n".join(content))
        f.write("\n")

with open("imdb62.txt") as f:
    data = [line.split("\t") for line in f.read().strip("\n").splitlines()]

# Check data integrity
for review in data:
    assert(len(review) == 6), review

authors = list(set([review[1] for review in data]))
author2id = {author: [] for author in authors}
feature_vec = np.zeros((len(data), len(authors)))
content = []
paper_ids = []

for i, review in enumerate(data):
    author2id[review[1]].append(review[0])
    paper_ids.append(review[0])
    feature_vec[i, authors.index(review[1])] = 1

    # Title + Content
    content.append(review[-2] + " " + review[-1])

writeToFile(authors, feature_vec, author2id, paper_ids, content)

