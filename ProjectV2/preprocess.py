import nltk
import re
""" These functions are used to replace tweet slang by words which are more easy
to be recognized by the sentiment analysis and to reduce the number of dimensions of
the features dictionnary.  """


# dictionnary to sentiment analysis
emo_repl = {
    #goodemotions
    "&lt;3": "good",
    ":d": "good",
    ":dd": "good",
    ":p": "good",
    "8)": "good",
    ":-)": "good",
    ":')": "good",
    ":)": "good",
    ";)": "good",
    "(-:": "good",
    "(:": "good",
    ":D": "good",
    ":P": "good",

    "yay!": "good",
    "yay": "good",
    "yaay": "good",
    "yaaay": "good",
    "yaaaay": "good",
    "yaaaaay": "good",
    #bademotions
    ":/": "bad",
    ":&gt;": "sad",
    ":'(": "sad",
    ":-(": "bad",
    ":(": "bad",
    ":s": "bad",
    ":-s": "bad"
}

# dictionnary for general (i.e. topic modeler)
emo_repl2 = {
    #goodemotions
    "&lt;3": "heart",
    ":d": "smile",
    ":p": "smile",
    ":dd": "smile",
    "8)": "smile",
    ":-)": "smile",
    ":')": "smile",
    ":)": "smile",
    ";)": "smile",
    "(-:": "smile",
    "(:": "smile",

    # bad emotions
    ":/": "worry",
    ":&gt;": "angry",
    ":'(": "sad",
    ":-(": "sad",
    ":(": "sad",
    ":s": "sad",
    ":-s": "sad"
}

# general
re_repl = {
    r"\br\b": "are",
    r"\bu\b": "you",
    r"\bhaha\b": "ha",
    r"\bhahaha\b": "ha",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhadn't\b": "had not",
    r"\bwon't\b": "will not",
    r"\bwouldn't\b": "would not",
    r"\bcan't\b": "can not",
    r"\bcannot\b": "can not",
    r"\'ve\b": " have",
    r"\'ll\b": " will",
    r"\'re\b": " are",

}

emo_repl_order = [k for (k_len, k) in reversed(sorted([(len(k), k) for k in emo_repl.keys()]))]
emo_repl_order2 = [k for (k_len, k) in reversed(sorted([(len(k), k) for k in emo_repl2.keys()]))]

#Replace the regular emoticons
def replace_emo(sentence):
    sentence2 = sentence
    for k in emo_repl_order:
        sentence2 = sentence2.replace(k, emo_repl[k])
    for r, repl in re_repl.items():
        sentence2 = re.sub(r, repl, sentence2)
    return sentence2

#Replace regularly troublesome parts of speech
def replace_reg(sentence):
    sentence2 = sentence
    for k in emo_repl_order2:
        sentence2 = sentence2.replace(k, emo_repl2[k])
    for r, repl in re_repl.items():
        sentence2 = re.sub(r, repl, sentence2)
    return sentence2

