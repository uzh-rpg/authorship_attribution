""" Summary
This script parses the metainformation to generate a dataset. First the
   arxiv-metadata-oai-snapshot.json
file from the arXiv dataset is parsed. The author names are extracted and
only authors who gave a full (>= 2 letters, no initials like L.B.) first
and lastname are considered. This is to reduce the author ambiguity as
there is no unique author ID in the dataset.

Now a list of unique authors is created and all authors from this list are
chosen that have more than N publications on arXiv. For the D500 dataset,
for example, N=500 and for the D100, N=100.

Once done, the script creates the files
    dataset/abstracts.csv
    dataset/author2id.csv
    dataset/features.csv
    dataset/paper_ids.csv
    dataset/authors.txt
"""

import os
import re
import sys
import json
import time
import numpy as np
import pandas as pd
import pickle as pkl
from sortedcontainers import SortedList


def read_in(basepath):
    """ READ IN
      " Read the whole file into a list of dictionaries to be then fed into
      " Pandas as a dataframe for further processing.
    """
    infile = open(basepath + "arxiv-metadata-oai-snapshot.json", "r")
    data_lod = []
    for line in infile:
        entry = json.loads(line)

        """ Extract meaningful names and initials from the authors_parsed field
          " 0. Check if the author has a first and a last name
          " 1. Get first and last name
          " 2. Check that the name parts contains letters, else continue
        """
        valid = True
        processed_authors = []
        author_list = entry["authors_parsed"]
        for author in author_list:
            # If the author has less than two fields, consider it invalid
            if len(author) < 2:
                continue
            lastname = author[0]
            firstname = author[1].split(" ")

            # Regex matching to get initials
            match=re.search(r"\w", firstname[0])
            if match:
                initials = match.group(0)
                for name in firstname[1:]:
                    group=re.search(r"\w", name)
                    if group:
                        initials = initials + " " + group.group(0)

                # Check if the firstname is an initial or a name
                if len(firstname[0])>1 and firstname[0][1].isalpha():
                    firstname = firstname[0]
                else:
                    firstname = None

                processed_authors.append([lastname, firstname, initials])
            else:
                continue

        parsed_entry = {}
        if processed_authors != []:
            parsed_entry["id"] = entry["id"]
            parsed_entry["title"] = entry["title"]
            parsed_entry["cat"] = entry["categories"]
            parsed_entry["authors"] = processed_authors
            parsed_entry["abstract"] = entry["abstract"]
        else:
            continue

        data_lod.append(parsed_entry)

    data = pd.DataFrame(data_lod)
    data.set_index(data["id"], inplace=True)
    pkl.dump(data, open("data_df.pkl", "wb"))
    return data


def connect_authors_ids(data):
    """ Associate authors and papers
    After filtering the author names, two associations need to be set up:
    id2author: Dictionary: paper id -> author name
    author2id: Dictionary: author name -> paper id
    """
    use_initials = False
    max_authors = 20
    id2author = {}
    author2id = {}
    for idx, paper in data.iterrows():
        count = 0
        if len(paper["authors"]) > max_authors:
            continue
        for author in paper["authors"]:
            if not use_initials:
                if author[1] is None:
                    continue
            key = author[0] + " " + (author[2] if use_initials else author[1])
            try:
                author2id[key].append(paper["id"])
            except KeyError:
                author2id[key] = [paper["id"]]
            if count == 0:
                id2author[paper["id"]] = [key]
            else:
                id2author[paper["id"]].append(key)
            count += 1

    pkl.dump(id2author, open("id2author.pkl","wb"))
    pkl.dump(author2id, open("author2id.pkl","wb"))
    return id2author, author2id


def threshold_authors(threshold = 500):
    """ Find authors with at least *threshold* papers in the dataset
    Return:
    author2id: Dictionary: author name -> paper ids
    """
    del_keys = []
    for key in author2id.keys():
        if len(author2id[key]) < threshold:
            del_keys.append(key)

    for key in del_keys:
        del author2id[key]

    pkl.dump(author2id, open("author2id_threshold.pkl","wb"))
    return author2id


def write_dataset(data, author2id):
    """ Creates Label Matrix and writes everything to the disk
    Creates a label matrix "feature_vec" according to the specification
    laid out in the main README.md (Section "Data Format")
    """

    authors = list(author2id.keys())

    # Step 4: Create one-hot vectors for the files
    print("Creating feature vector")
    paper_list = []
    for author in authors:
        paper_list.extend(author2id[author])
    paper_list = SortedList(list(set(paper_list)))
    feature_vec = np.zeros((len(paper_list), len(authors)))
    print("   step one is done")

    for author in authors:
        ids = author2id[author]
        idx_author = authors.index(author)
        for paper_id in ids: #entry in data["title"][ids]:
            idx_paper = paper_list.index(paper_id)
            feature_vec[idx_paper, idx_author] = 1

    paper_list = list(paper_list)

    print("Extracting Abstracts")
    paper_set = set(paper_list)
    mask = [x in paper_set for x in data.index]
    mask = pd.array(mask, dtype=bool)
    print("    masking done")
    abstract_df = data[mask]["abstract"]
    tmp = []
    for paper_id in paper_list:
        tmp.append(abstract_df.loc[paper_id]\
                .replace("\n"," ").replace("\x0c"," "))
    s = "\n".join(tmp) + "\n"
    print("    extraction done")


    with open("dataset/abstracts.csv", "w") as f:
        f.write(s)
    print("Saving files")
    np.savetxt("dataset/features.csv", feature_vec, fmt="%1i")
    np.savetxt("dataset/authors.txt", authors, fmt="%s")
    np.savetxt("dataset/paper_ids.csv", paper_list, fmt="%s")
    np.savetxt("dataset/author2id.csv", np.array(author2id[authors[0]], dtype=str, ndmin=2), fmt="%s", newline='')
    with open("dataset/author2id.csv", "ab") as f:
        f.write(b"\n")
        for i in range(1, len(authors)):
            np.savetxt(f, np.array(author2id[authors[i]], dtype=str, ndmin=2), fmt="%s")


""" Main Function to run the script
The REDO list influences which of the above steps are done again.
This saves plenty of time when debugging the script. All intermediate
results are saved to pickle files to resume the script at any point.
"""
if __name__=="__main__":
    #  REDO = [False, False, False, True, True]
    REDO = [True, True, True, True, True]

    #  Step 1: data loader
    if (REDO[0]):
        if len(sys.argv) == 1:
            data = read_in("../0_resources/")
        else:
            data = read_in(sys.argv[1])
    else:
        if (REDO[1]):
            data = pkl.load(open("data_df.pkl","rb"))
        else:
            pass
    print("Finished loading the data")

    # Step 2: create unique list of authors
    if (REDO[1]):
        id2author, author2id = connect_authors_ids(data)
    else:
        if (REDO[2]):
            id2author = pkl.load(open("id2author.pkl","rb"))
            author2id = pkl.load(open("author2id.pkl","rb"))
        else:
            pass
    print("Finished creating list of unique authors")


    # Step 3: threshold minimum number of authors
    if (REDO[2]):
        author2id = threshold_authors(threshold = 500)
    else:
        if (REDO[3]):
            author2id = pkl.load(open("author2id_threshold.pkl","rb"))
        else:
            pass
    print("Finished filtering authors")

    # Step 4: Write to CSV file
    if (REDO[3]):
        if 'data' not in locals():
            data = pkl.load(open("data_df.pkl","rb"))
        if 'author2id' not in locals():
            author2id = pkl.load(open("author2id_threshold.pkl","rb"))
        write_dataset(data, author2id)
    else:
        pass
    print("Finished write to csv")


