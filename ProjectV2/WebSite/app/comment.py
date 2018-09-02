from textblob import TextBlob


# Comment object to store required information from the file
class Comment:
    def __init__(self, text):
        self.text = text
        self.split_text = text.split()
        self.blob = TextBlob(self.text)
        self.words = self.blob.words

    def __iter__(self):
        for i in range[0:len(self.split_text) - 1]:
            return (self.split_text[i])

    def __next__(self):
        if i < len(self.split_text) - 1:
            return self.split_text[i]
        else:
            raise StopIteration  # Done iterating.

    def __array__(self):
        result = [self.text]
        return result

    def getInfo(self):
        print("Text: ", self.text)

