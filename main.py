import nltk
import re
import pandas as pd

def print_dataset(fileName, n=100):
    with open(fileName, 'rb') as datafile:
        lines = datafile.readlines()
        for line in lines[:n]:
            print(line)
def load_lines(fileName, fields = ["lineID", "characterID", "movieID", "character", "text"]):
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj
    return lines
# print_dataset('cornell movie-dialogs corpus/movie_lines.txt')
dataset = load_lines('cornell movie-dialogs corpus/movie_lines.txt')

data = pd.DataFrame(dataset).T
print(data[data['movieID'] == 'm0'])

