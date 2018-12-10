import torch
import torch.nn as nn
import torch.nn.functional as F

from nltk.stem import PorterStemmer
import unicodedata
import re
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 25


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, 1)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = self.out(output[0])
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

SOS_token = 0
EOS_token = 1

stemmer = PorterStemmer()


class Text:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        word = stemmer.stem(word)
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def readTexts(input_filename, target_filename):
    with open(input_filename, encoding='utf-8') as f:
        lines = f.read().strip().split('\n')

    with open(target_filename) as f:
        targets = f.read().strip().split('\n')

    lines = [normalizeString(l) for l in lines]
    targets = [float(t) for t in targets]
    return Text(), list(zip(lines, targets))


def filterLine(l):
    return len(l.split(' ')) < MAX_LENGTH

def filterLines(lines):
    return [line for line in lines if filterLine(line[0])]


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def prepareData(input_filename, target_filename):
    text, lines = readTexts(input_filename, target_filename)
    print("Read %s sentence lines" % len(lines))
    lines = filterLines(lines)
    print("Trimmed to %s sentence lines" % len(lines))
    print("Counting words...")
    for line in lines:
        text.addSentence(line[0])
    print("Counted words:")
    print(text.n_words)
    return text, lines


# text, lines = prepareData(filename)
# print(random.choice(lines))


'''
Convert to Tensors
'''

def indexesFromSentence(lang, sentence):
    return [lang.word2index[stemmer.stem(word)] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorFromTarget(target):
    return torch.tensor(target, dtype=torch.float, device=device).view(-1, 1)


def tensorsFromLine(text, line):
    input_tensor = tensorFromSentence(text, line[0])
    target_tensor = tensorFromTarget(line[1])
    return (input_tensor, target_tensor)
