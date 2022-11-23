# Introduction
### Data Format
This section specifies the data format chosen in this work. The representation was chosen because it is human-readable and human-interpretable as well as being very easy to parse. In total, 7 different files exist. Note that the file-ending `.txt` or `.csv` carries no meaning and is only present for legacy-reasons.

In the following, assume that the dataset contains *N* authors and *P* papers. The notation *(a x b [c])* means that a file has *a* lines *b* columns seperated by character *c*. If a number is unspecified or does not apply, it is replaced with a *%*.

##### `authors.txt`  (N x 1)
A list of all authors in the dataset. The *n*-th author is defined as the *n*-th line in this file.

##### `paper_ids.csv` (P x 1)
A list of all paper ids in the dataset. The *p*-th paper is defined as the *p*-th line in this file. A paper id may not contain whitespaces and must be unique for all papers in the dataset.

##### `author2id.csv` (N x % [' '])
This file contains the associations between the authors and the papers in a human readable form. The *n*-th line represents the *n*-th author and it contains a white-space seperated list of paper ids. That is to say that the *n*-th author was an author for all papers listed in the *n*-th line of this file.

##### `features.csv` (P x N [' '])
The label matrix. Each of the *P* lines contains exactly *N* numbers, either 0 or 1. If the matrix entry (i,j) is 1, it means that the *j*-th author is an author of the *i*-th paper.

##### `content.csv` (P x 1)
This file contains the main body of the article (i.e. this is fed into the transformer to predict the labels). Each line contains the entire content (minus newlines, obviously) of the *p*-th article.

##### `abstracts.csv` (P x 1)
This file contains the abstract of the article, where the *p*-th line contains the abstract of the *p*-th article.

##### `references.csv` (P x 1)
This file contains the reference section of the article, where the *p*-th line contains the reference section of the *p*-th article.

##### `reference_authors.csv` (P x % ['####', '|'])
This file contains the names of the authors of the references/citations from the references.csv file. In general this splitting is sometimes completely off, but works amazingly well most of the time. The `####` delimiter segments the individual references (e.g. if a paper has 14 references, then there will be 13 such delimiters). The `|` delimiter split the author names per reference, so that this paper may look like `bauersfeld|romero|muglikar`. Note all authors are lowercased. 

##### `reference_count_embeddings.csv` (P x % [' '])
This file contains the histogram embeddings for the authors' names of the cited papers. 

##### `split_index.txt` (1 x 1)
A text file containing the (zero based) index *i* where the train/test split is divided. The paper in line *P=i-1* is the first in the test set.

##### `split_id.txt` (1 x 1)
A text file containing the arXiv ID of the first paper in the test set. Can be used to validate the index.


# Running the Code
### The raw arXiv Dataset
To be able to run the scripts contained in the `1_dataset` folder, one needs the arXiv dataset. This can be downloaded and pre-processed with the open-source scripts: https://github.com/mattbierbaum/arxiv-public-datasets.

In the following it is assumed that a user performs all steps from their Readme up to (and including) the section "Plain text".

Create a folder `0_resources` and link the contents there from the downloaded dataset
```
$ mkdir 0_resources
$ cd 0_resources
$ ln -s PATH_TO_ARXIV_DATA/arxiv-metadata-oai-snapshot.json .
$ ln -s PATH_TO_ARXIV_DATA/arxiv .
```
If done correctly, the output should look like this:
```
$ cd 0_resources
$ ls
arxiv  arxiv-metadata-oai-snapshot.json
$ cd arxiv
$ ls
fulltext  output  tarpdfs
```

### Settings up the Python Enviroment
To run the code for this project, one needs to install a couple python packages. A conda environment can be used which is based on `dl2021.yml` also found in the toplevel directory of this project.
```
$ conda env create -f dl2021.yml
```

Alternatively, a `venv` can be used with the help of the provided file `requirements.txt` provided in the top-level folder of this project (we don't guarantee that this always works).
```
$ python3 -m venv dl2021
$ source dl2021/bin/activate
$ python -m pip install -r requirements.txt
```

All commands presented below assume that the created enviroment has been activated, i.e. on of the following commands has been executed.
```
$ conda activate dl2021
$ source dl2021/bin/activate
```


### Creating the Dataset

Create a folder `1_dataset/dataset` before running the dataset creation scripts. The threshold of how many papers a specific author must have authored to be part of the dataset can be set in line 199 of `1_dataset/1_create_dataset.py`.
```
$ cd 1_datset
$ mkdir dataset
$ python 1_create_dataset.py
$ python 2_document_parser.py
$ python 3_postprocess_data.py
```
Use vim commands `:%s/\s\+/ /g` and `:%s/^\s\+//g` on the `dataset/author2id.csv` file to remove some whitespaces (just much faster than python).
```
$ python 4_seperate_authors.py
$ python 5_remove_bad_authors.py
$ python 6_train_test_split.py
$ python 7_references_names.py
```
Optionally, run the chunking script to divide the content of the papers into chunks of T words (T=512, by default). This increases the amount of training data per author as the transformer architecture DistilBERT can only take the first 512 words as an input.
```
$ python 8_split_content.py
```

The scripts perform the following tasks:
1. Parse the `arxiv-supplied arxiv-metadata-oai-snapshot.json` file to extract information about the dataset. This script creates the files
  `authors.txt`,
  `paper_ids.csv`,
  `author2id.csv`,
  `features.csv`
2. For all papers selected, extract the abstract, content and reference section. This script creates the files
  `content.csv`,
  `abstract.csv`,
  `references.csv`
3. Filter out all papers from the dataset, where the second script failed to extract meaningful information.
4. Because arXiv has no unique author identifier, there author ambiguities (e.g. same name, different person). Based on a simple clustering of the abstract, the script `4_seperate_authors.py` tries to guess which name is uniquely associated to a person and where ambiguitites occur. The script generates the `bad_authors.csv` file which contains the ids of all papers where the author is not unique.
5. Clean up the dataset again by removing ambigious authors.
6. Create a randomly shuffled training and a test split, where a split of about 80/20 is ensured _for each author_. This is important as just splitting the dataset randomly might result in an uneven distribution per author. Furthermore, if multiple persons co-authored a paper, it is ensured that it can not be in the training set for one person but not for the other. Lastly, co-authored papers in the training set are _randomly assigned_ to one of their co-authors. The script produces
  `split_index.txt`
  `split_id.txt`
7. Extract the names from the references.   
8. The scipt divides the content of each paper into chunks of size T=512. This increases the size of the training and evaluation set because DistilBERT can only process 512 words and without this final step thus only the start of each paper would be used.

For further information regarding the scripts, please see the "summary section" at the beginning of each script.


### Architectures
In this section we describe how to run the architectures described in the paper. There are further parameters and instructions in the comments inside each of the source files. Please read them for a proper guide on how to use these files.

#### distilBERT (Content + References):
The distilBERT content and references network can be trained by the following commands:
```
cd 2_distilbert
python train_custom_distilbert.py
```

#### fastText embedding:
To train the fastText and create the fasText embeddings:

```
cd 3_fasttext
python create_fast_text_embedding.py

```

#### References Only:
The references network can be trained by the following commands:
```
cd 4_train_only_refs
python train_only_refs_distilbert.py
```


#### distilBERT (Content only):
The distilBERT content only network can be trained by the following commands:
```
cd 5_train_only_content
python train_only_content_distilbert.py
```

#### LSTM 

The LSTM architecture for references is in the folder `6_train_lstm_only_refs` and the combination of LSTM with BERT Classification is in folder `7_train_lstm_refs_and_content`

To train the reference only architecture with LSTM run the following command:
```
cd 6_train_lstm_only_refs
python train_lstm.py
```

To train the reference and context architecture with LSTM run the following command:
```
cd 7_train_lstm_refs_and_content
python train_lstm_bert.py
```

### Baselines (9_baseline_dataset)
To compare our approach with published baselines, we will use the "Reuters-50-50" (aka CCAT50) dataset in this tutorial, available at https://archive.ics.uci.edu/ml/datasets/Reuter_50_50. The scripts in the `9_baseline_dataset` folder convert this classic authorship-identification dataset into the standard form described above.

To run the code, first create the output folders which will contain the converted dataset. The `dataset_50_50` folder retains the original train/test split of the Reuters_50_50 dataset, whereas the `dataset-reuters` folder merges the train/test data to enable different splits (e.g. 90/10).
```
$ cd 9_baseline_dataset
$ cd reuters
$ mkdir dataset_50_50 dataset-reuters
```

Then run the two python scripts
```
$ python 1_create_dataset.py
$ python 2_merge_dataset.py
$ python 3_train_test_split.py
$ python 4_split_content.py
```
For details regarding the first two scripts, please see their header section. The two last ones are basically a copy from the `1_dataset` folder.

The structure for the remaining datasets is similar. They can be downloaded from the internet

Blog: https://metatext.io/datasets/blogger-authorship-corpus

IMDb62: https://umlt.infotech.monash.edu/?page_id=266

Legal/Judgement: https://umlt.infotech.monash.edu/?page_id=152

