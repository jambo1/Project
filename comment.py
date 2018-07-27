from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Comment object to store required information from the file
class Comment:
    def __init__(self, text):
        # Isn't necessary to save the label
        # self.label = label
        self.text = text
        self.split_text = text.split()
        self.blob = TextBlob(self.text)
        self.words = self.blob.words
        self.sentences = self.blob.sentences


        laughing = ["lmao", "lmfao", "pmsl", "rofl", "lol"]
        count = 0
        for word in self.words:
            if word in laughing:
                count += 1

        self.lols = count

    def __iter__(self):
        for i in range[0:len(self.split_text) - 1]:
            return (self.split_text[i])

    def __next__(self):
        if i < len(self.split_text) - 1:
            return self.split_text[i]
        else:
            raise StopIteration  # Done iterating.

    def __array__(self):
        result = [self.label, self.text, self.lols]
        return result

    def getInfo(self):
        print("Label: ", self.label, " Text: ", self.text)

