# Open the file and create comment objects with the label and comment

import csv
from app.preprocess import replace_emo, replace_reg
from app.comment import Comment
import numpy as np

def read_dirty_file(filename):
    sarc_comments = []
    neg_comments = []

    with open(filename, encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if len(row['comment'].split())>3:
                sentence = row['comment']
                #sentence = replace_emo(sentence)
                sentence = replace_reg(sentence)
                if(row['label']=='1'):
                    sarc_comments.append(Comment(sentence))
                elif(row['label']=='0'):
                    neg_comments.append(Comment(sentence))

    #Put both into np arrays so they can be easier accessed for future uses
    sarc_coms = np.array(sarc_comments)
    neg_coms = np.array(neg_comments)
    np.save('sarccoms', sarc_coms)
    np.save('negcoms', neg_coms)

    #Return both sets of comments
    return sarc_comments, neg_comments

def read_sarc_file(filename):
    sarc_comments = []

    with open(filename, encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if len(row['comment'].split())>3:
                sentence = row['comment']
                #sentence = replace_emo(sentence)
                sentence = replace_reg(sentence)
                if(row['label']=='1'):
                    sarc_comments.append(Comment(sentence))

    #Put both into np arrays so they can be easier accessed for future uses
    sarc_coms = np.array(sarc_comments)
    np.save('sarccoms', sarc_coms)

    #Return both sets of comments
    return sarc_comments

def read_neg_file(filename):
    neg_comments = []

    with open(filename, encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if len(row['comment'].split())>3:
                sentence = row['comment']
                #sentence = replace_emo(sentence)
                sentence = replace_reg(sentence)
                if(row['label']=='0'):
                    neg_comments.append(Comment(sentence))

    #Put both into np arrays so they can be easier accessed for future uses
    neg_coms = np.array(neg_comments)
    np.save('negcoms', neg_coms)

    #Return both sets of comments
    return neg_comments