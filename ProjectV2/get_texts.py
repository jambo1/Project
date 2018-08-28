# Open the file and create comment objects with the label and comment

import csv
from ProjectV2.preprocess import replace_emo, replace_reg
from ProjectV2.comment import Comment
import numpy as np

#Takes a CSV file as input and returns arrays of sarcastic, non-sarcastic or an array of each type of comments
def read_file(filename):
    sarc_comments = []
    neg_comments = []
    #Open CSV file and read each line
    with open(filename, encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            #Must have more than 3 words
            if len(row['comment'].split())>3:
                count=0
                #Count the alphanumeric words
                for word in row['comment'].split():
                    if word.isalnum():
                        count += 1
                #If there are 3 or more alphanumeric words then add it to the relevant array
                if count >= 3:
                    sentence = row['comment']
                    sentence = replace_reg(sentence)
                    if(row['label']=='1'):
                        sarc_comments.append(Comment(sentence))
                    elif(row['label']=='0'):
                        neg_comments.append(Comment(sentence))

    #If both arrays have elements save both into np arrays so they can be easier accessed for future uses and return both
    if len(sarc_comments) >0 and len(neg_comments) >0:
        sarc_coms = np.array(sarc_comments)
        np.save('sarccoms', sarc_coms)
        neg_coms = np.array(neg_comments)
        np.save('negcoms', neg_coms)
        # Return both sets of comments
        return sarc_comments, neg_comments

    #If only sarcastic comments then save this and return them
    elif len(sarc_comments) >0:
        sarc_coms = np.array(sarc_comments)
        np.save('sarccoms', sarc_coms)
        #return sarc_comments
        return sarc_comments

    #If only non sarcastic then save and return them
    elif len(neg_comments) >0:
        neg_coms = np.array(neg_comments)
        np.save('negcoms', neg_coms)
        #return neg_comments
        return neg_comments



