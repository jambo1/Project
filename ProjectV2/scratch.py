import ProjectV2.get_texts as get_texts
from ProjectV2.comment import Comment

sarc_comments, neg_comments = get_texts.read_file("tests.csv")
print(len(neg_comments),",",len(sarc_comments))