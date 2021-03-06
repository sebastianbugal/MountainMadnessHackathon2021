import nltk
import re
import pandas as pd
import numpy as np

def main():
    pairs = process_data()
    in_token = find_tokens(pairs)['in_token']
    out_token = find_tokens(pairs)['out_token']
    print(make_feature_dic(in_token, out_token)['reverse_output_dic'])

def make_feature_dic(in_token, out_token):
    input_dic = dict(
        [(token, i) for i, token in enumerate(in_token)])
    output_dic = dict(
        [(token, i) for i, token in enumerate(out_token)])

    reverse_input_dic = dict(
        (i, token) for token, i in input_dic.items())
    reverse_output_dic = dict(
        (i, token) for token, i in output_dic.items())

    return {
        'input_dic': input_dic,
        'output_dic': output_dic,
        'reverse_input_dic': reverse_input_dic,
        'reverse_output_dic': reverse_output_dic
    }

def find_tokens(pairs):
    input_arr, output_arr = [], []
    in_token = set()
    out_token = set()

    for i in pairs:
        input, output = i['in'], i['out']
        input = input.lower()
        output = output.lower()
        input_arr.append(input)
        output = " ".join(re.findall(r"[\w']+|[^\s\w]", output))
        output = '<START> ' + output + ' <END>'
        output_arr.append(output)

        for tok in re.findall(r"[\w']+|[^\s\w]", input):
            if tok not in in_token:
                in_token.add(tok)
        for tok in output.split():
            if tok not in out_token:
                out_token.add(tok)

    in_token = sorted(list(in_token))
    out_token = sorted(list(out_token))
    in_token_len = len(in_token)
    out_token_len = len(out_token)

    return {'in_token': in_token, 'out_token': out_token, 'input_arr': input_arr, 'output_arr': output_arr, 'in_token_len': in_token_len, 'out_token_len': out_token_len}


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


def process_data():
    pairs = []
    men = ['u0', 'u7133', 'u667', 'u695', 'u6937', 'u861', 'u7366', 'u6912', 'u1755', 'u3784', 'u2010', 'u3819', '7384',
           'u2495', 'u6378', 'u2511']

    movieID_array = ['m0', 'm479', 'm42', 'm43', 'm464', 'm55',
                     'm498', 'm462', 'm116', 'm249', 'm130',
                     'm252', 'm499', 'm161', 'm425', 'm163']
    df = pd.read_csv('MovieSetDataFrame.csv')
    df = df[::-1]
    df = df.reset_index(drop=True)
    df['lineID'] = df['lineID'].apply(lambda x: re.sub(r'[a-zA-Z]', '', str(x)))
    df['lineID'] = pd.to_numeric(df['lineID'])
    for i in movieID_array:
        temp = df[df['movieID'] == i]
        man = None
        for row in temp.iterrows():
            if row[1]['characterID'] in men:
                if man is None:
                    man = row
                elif man[1]['characterID'] == row[1]['characterID']:
                    man = row
            elif man:
                if man[1]['lineID']-1 == row[1]['lineID'] or man[1]['lineID']+1 == row[1]['lineID']:
                    pairs.append({'in': man[1]['text'],'out': row[1]['text']})
    return pairs

if __name__ == "__main__":
    main()



















































# print_dataset('cornell movie-dialogs corpus/movie_lines.txt')
dataset = load_lines('cornell movie-dialogs corpus/movie_lines.txt')
# df = pd.DataFrame()
# data = pd.DataFrame(dataset).T

# df = df.append(data[(data['movieID'] == 'm0') & ((data['characterID'] == 'u0') | (data['characterID'] == 'u2'))])
# df = df.append(data[(data['movieID'] == 'm479') & ((data['characterID'] == 'u7133') | (data['characterID'] == 'u7142'))])
# df = df.append(data[(data['movieID'] == 'm42') & ((data['characterID'] == 'u667') | (data['characterID'] == 'u664'))])
# df = df.append(data[(data['movieID'] == 'm44') & ((data['characterID'] == 'u695') | (data['characterID'] == 'u706'))])
# df = df.append(data[(data['movieID'] == 'm464') & ((data['characterID'] == 'u6937') | (data['characterID'] == 'u6940'))])
# df = df.append(data[(data['movieID'] == 'm55') & ((data['characterID'] == 'u861') | (data['characterID'] == 'u859'))])
# df = df.append(data[(data['movieID'] == 'm498') & ((data['characterID'] == 'u7366 ') | (data['characterID'] == 'u7367'))])
# df = df.append(data[(data['movieID'] == 'm462') & ((data['characterID'] == 'u6912 ') | (data['characterID'] == 'u6896'))])
# df = df.append(data[(data['movieID'] == 'm116') & ((data['characterID'] == 'u1755') | (data['characterID'] == 'u1758'))])
# df = df.append(data[(data['movieID'] == 'm249') & ((data['characterID'] == 'u3784') | (data['characterID'] == 'u3777'))])
# df = df.append(data[(data['movieID'] == 'm130') & ((data['characterID'] == 'u2010') | (data['characterID'] == 'u2004'))])
# df = df.append(data[(data['movieID'] == 'm252') & ((data['characterID'] == 'u3819') | (data['characterID'] == 'u3820'))])
# df = df.append(data[(data['movieID'] == 'm499') & ((data['characterID'] == 'u7384') | (data['characterID'] == 'u7372'))])
# df = df.append(data[(data['movieID'] == 'm161') & ((data['characterID'] == 'u2495') | (data['characterID'] == 'u2485'))])
# df = df.append(data[(data['movieID'] == 'm425') & ((data['characterID'] == 'u6378') | (data['characterID'] == 'u6379'))])
# df = df.append(data[(data['movieID'] == 'm163') & ((data['characterID'] == 'u2511') |(data['characterID'] == 'u2526') | (data['characterID'] == 'u2521'))])


