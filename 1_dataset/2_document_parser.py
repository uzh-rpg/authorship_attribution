""" Summary
This script reads the abstracts, content, and references for all papers.
It opens all the textfiles from the PDF-to-text converstion process. Then,
the individual sections are found using some regular expressions.

If no main section or a reference section is found, the corresponding
entry is left blank and the paper will be discarded in a subsequent step
(not within this script).

Once done, the script creates the files
    dataset/content.csv
    dataset/references.csv


Note on the regular expression-based splitting.  Assumed structure:
    HEADER
    CONTENT
    REFERENCES
    STUFF (Some more stuff like Appendix, List of Figures, ...)_
The References can be started by the word "References" (and variations of it). 
The References are either terminated by the end of file (EOF) or by an appendix 
section ("Appendix", "Appendices", "Supplement", "Additional", and variations). 
Also, a [1] at the beginning of a line can start the reference section if none 
of the keywords above matches.

The header is started by the beginning of the file (BOF). It is terminated by
the "Abstract" (and variations).
The content of the paper is everything between the header and the references.

A sanity check on the lengths of the header, content and references is,
performed and offenders are removed by the next script.
"""

import os
import re
import sys
import time
import numpy as np
import pandas as pd
from uniplot import plot
from collections import Counter

""" Regex to split a document """
re_content_begin = re.compile(r"^\s?A\s?bstract", re.IGNORECASE | re.MULTILINE)

re_references_begin = [
    re.compile(r"^\s?R\s?(?i:eferences)", re.MULTILINE),
    re.compile(r"^\s?\[1\]", re.MULTILINE)
]

re_stuff_begin = [
    re.compile(r"^\s?A\s?ppendi", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s?(Online\s?)?S\s?upplement", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s?A\s?dditional", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s?E\s?xperiment", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s?F\s?ig\.?\s?\d",re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s?T\s?ab\.?\s?\d",re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s?F\s?igure",     re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s?T\s?able",      re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s?D\s?iscussion", re.IGNORECASE | re.MULTILINE)
    ]


remove_blanklines = re.compile(r"^\s*$\n", re.MULTILINE)
remove_emaillines = re.compile(r"^.*\@.*$",re.MULTILINE)
remove_numberlines = re.compile(r"^\d*\.?\d*$", re.MULTILINE)

class Logger:
    """
    Simple logger class for debugging. Doesn't do anything by default and 
    writes a list of papers that did not make the cut (i.e. splitting 
    failed) to a file in the dataset folder.
    """
    def __init__(self, LOG=False):
        self.error_list = []
        self.LOG = LOG

    def log(self, name, reason):
        self.error_list.append([name, reason])

    def write(self):
        if self.LOG:
            with open("dataset/failed.txt", "w") as f:
                for x in self.error_list:
                    f.write("%20s\t%s\n" % (x[0], x[1]))


def split_references(f, references):
    """ Function to split a reference section into single references.
    Logic: 5 criteria are checked if they make sense (if one fails,
    the next one is used. If one matches, it is used without checking
    the rest)
    References all
    1. have a "." at the end
    2. have a digit at the end, e.g. page number, year
    3. have a ")" at the end, e.g. (year)
    4. start with a "["
    5. start with a digit followed by a "."

    A criteria is considered a match if more than 1/4 of all lines 
    in the section match it. This means, the average reference must
    be shorter than 4 lines. Also, the criteria needs to match at 
    least a couple number of times (that's different for each type)

    Based on the splitting references are identified. Sometimes,
    plain text is fasly segmented into the references section. Thus
    to be considered a reference, it needs to contain at least 10
    alphabetic characters and at least 7% special characters, which
    are .,"-:

    In the end, the references are returned as a joint string each
    reference seperated by ####
    """
    # Remove too short lines first
    references = references.replace("\x0c","\n")
    references = "\n".join([x for x in references.splitlines() if len(x.strip()) > 5]) + "\n"
    re.sub(r"^\s*.{,5}\s*$", "", references, re.MULTILINE)
    number_dotlines = len(re.findall(r"\.\s*$", references, re.MULTILINE))
    number_numberlines = len(re.findall(r"\d\s*$", references, re.MULTILINE))
    number_parlines = len(re.findall(r"\)\s*$", references, re.MULTILINE))
    number_bracketlines = len(re.findall(r"^\s*\[", references, re.MULTILINE))
    number_countlines = len(re.findall(r"^\s*\d+\.", references, re.MULTILINE))
    number_lines = references.count("\n")
    if (number_dotlines/number_lines > 0.25 and number_dotlines > 3) \
            or number_dotlines >= 10:
        lines = references.strip("\n").split(".\n")
    elif number_numberlines/number_lines > 0.25 and number_numberlines > 4:
        lines = re.split(r"\d\s*\n", references.strip("\n"))
    elif number_parlines/number_lines > 0.25 and number_parlines > 4:
        lines = re.split(r"\)\s*\n", references.strip("\n"))
    elif number_bracketlines/number_lines > 0.25 and number_bracketlines > 1 \
            or number_bracketlines >= 5:
        lines = re.split(r"\n\s*\[", references.strip("\n"))
    elif number_countlines/number_lines > 0.25 and number_countlines > 3 \
            or number_countlines >= 10:
        lines = re.split(r"\n\s*\[", references.strip("\n"))
    else:
        references = "#####\n" + references
        lines = references.splitlines()
    lines = [re.sub(r"^\[?\d+\.?\]?\s*", "", x) for x in lines]
    lines = [l.replace("\n"," ").replace("\x0c"," ") for l in lines]

    # Calculate statistics for each line
    chars = [len(re.findall(r"\w", x)) for x in lines]
    letters = [len(re.findall(r"[a-zA-Z]", x)) for x in lines]
    special_char = [len(re.findall(r"(?:\.|,|\"|-|:)", x)) for x in lines]
    special_char_ratio = [a/(b+1) for a,b in zip(special_char, chars)]

    #  lines = ["%5.2f %i %s" % (x,c,l) for x,c,l in zip(special_char_ratio, letters, lines)]
    lines = [l for x,c,l in zip(special_char_ratio, letters, lines) if x > 0.07 and c > 10]
    success = True
    if not lines:
        lines = ["ALL REFERENCES DELETED"]
        split_references.counter += 1
        success = False
    references = "####".join(lines)
    return references, success
split_references.counter = 0

def parse_papers(basedir, files, abstracts):
    assert(len(files) == len(abstracts)), "%i vs %i" % (len(files), len(abstracts))
    """ Performs the splitting discussed in the summary for given papers."""
    data = []
    lengths = []
    logger = Logger()


    count = 0
    t0 = time.time()
    for a,f in zip(abstracts,files):
        # Read in the file and remove blank lines 
        data.append(None)
        count += 1
        if not (count % 10000):
            print("%7i %6.4f" % (count, (time.time()-t0)/count))

        try:
            txt_file = open(os.path.join(basedir, f),'r')
        except:
            logger.log(f, "READ")
            continue
        try:
            content = txt_file.read()
        except UnicodeDecodeError:
            logger.log(f, "DECODE")
            continue
        content = remove_emaillines.sub('', content)
        content = remove_numberlines.sub('', content)
        content = remove_blanklines.sub('',content)

        # If word "references" is present, use it to split document
        success = False
        problem_name = ""
        for i, regex_begin in enumerate(re_references_begin):
            split = regex_begin.split(content)
            if len(split) > 1:
                references = split[-1] # Guess the last occurance is correct

                if i == 1: # Add [1] back to beginning
                    references = "[1]" + references

                # Trim it to remove the more stuff section
                for regex in re_stuff_begin:
                    references = regex.split(references)[0]

                # Check if the identified sectiom matches length requirements
                if len(references) > 20000 or len(references) < 50:
                    success = False
                    problem_name = "NO REF: ref length %i : %i" % (i, len(references))
                else:
                    success = True
                    content = split[0]
                    break

        if not success:
            logger.log(f, "NO REF" if problem_name == "" else problem_name)
            continue

        # Try to split away the header
        split = re_content_begin.split(content)
        if len(split) !=2 :
            try:
                split = re.split(a.strip()[:30], content, re.MULTILINE | re.IGNORECASE)
                split[1] = a.strip()[:30]+split[1]
            except Exception as e:
                logger.log(f, "ABS RE ERROR")
                continue

        if len(split) == 1:
            logger.log(f, "NO HEADER")
            continue

        # Splitting so far good
        header = split[0]
        content = split[1]
        if len(content) > len(references):
            references, success = split_references(f, references)
            if not success:
                logger.log(f, "SPLITTING FAILED")
                continue
            data[-1] = [header, content, references]
        else:
            logger.log(f, "CON < REF")

    t1 = time.time()
    logger.write()

    print((t1 - t0)/len(files))
    return data

""" Code that loops over all papers that are listed in the paper_ids.csv
file created by the previous script.
Once the content and reference section are extracted, they are stripped
of line-breaks (and page breaks \x0c). Then, they are written to the disk.
"""
if __name__=="__main__":
    if len(sys.argv) == 1:
        basepath = "../0_resources/arxiv/fulltext/arxiv"
    else:
        basepath = sys.argv[1]

    infile = open("dataset/paper_ids.csv", "r")
    paths = []
    for line in infile:
        item = line.strip()
        if "/" in item:
            category = item.split("/")[0]
            identifier = item.split("/")[1]
        else:
            category = ""
            identifier = item
        volume = identifier[:4]
        version = "v1"
        suffix = ".txt"
        path = os.path.join(basepath, category, "pdf",
                volume, identifier + version + suffix)
        paths.append(path)

    basedir = ""
    with open("dataset/abstracts.csv") as f:
        abstracts = f.read().strip("\n").split("\n")
    data = parse_papers(basedir, paths, abstracts)

    content_file = open("dataset/content.csv", "w")
    references_file = open("dataset/references.csv","w")
    for paper in data:
        if paper is None:
            pass
        else:
            header = paper[0].replace("\n"," ").replace("\x0c"," ")
            content = paper[1].replace("\n"," ").replace("\x0c"," ")
            references = paper[2]
            content_file.write(content)
            references_file.write(references)
        content_file.write("\n")
        references_file.write("\n")

    print(len(data))
    print(len([x for x in data if x is not None]))
    print(split_references.counter)
