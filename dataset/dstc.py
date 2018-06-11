import json
from collections import Counter
from enum import Enum
from pprint import pprint
from typing import Tuple

import nltk

class DSTC2:

    class Mode(Enum):
        train, dev, test = 0, 1, 2

    @classmethod
    def devset(cls, n_word):
        return cls(DSTC2.Mode.dev, n_word)

    @classmethod
    def trainset(cls, n_word):
        return cls(DSTC2.Mode.train, n_word)

    @classmethod
    def testset(cls, n_word):
        return cls(DSTC2.Mode.test, n_word)

    def __init__(self, mode: Mode, n_word):
        self.mode = mode
        self.label_json, self.log_json = list(self._filepath('label')), list(self._filepath('log'))
        self.word_dict, self.word_dict_rev = self._build_set(n_word)
        self.label_dict, self.label_dict_rev = self._build_label()

    def _filepath(self, which_one: str):
        dataset = self.mode.name
        with open('data/dstc2_{}/scripts/config/dstc2_{}.flist'.format(
                'test' if self.mode is DSTC2.Mode.test else 'traindev', dataset
        )) as flist:
            paths = flist.read().splitlines()
            for path in paths:
                path = 'data/dstc2_{}/data/'.format('test' if self.mode is DSTC2.Mode.test else 'traindev') + path + '/'
                with open(path + which_one + '.json') as f:
                    yield json.load(f)

    def read_json(self):
        utterances, labels = [], []
        for log in self.log_json:
            for turn in log['turns']:
                utterance = turn['output']['transcript']
                label = turn['output']['dialog-acts'][0]['act']
                utterances.append(utterance)
                labels.append(label)

        return utterances, labels

    def _build_set(self, n_words):
        counter = Counter()
        utterances, labels = self.read_json()
        for utterance in utterances:
            tokens = nltk.word_tokenize(utterance)
            counter.update(tokens)

        count = [['UNK', -1]]
        count.extend(counter.most_common(n_words - 1))

        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0
        for word in counter.most_common():
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count += 1
            data.append(index)
        count[0][1] = unk_count
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return dictionary, reversed_dictionary

    def _build_label(self):
        counter = Counter()
        _, labels = self.read_json()
        counter.update(labels)
        dictionary = dict()
        for i, word in enumerate(counter.most_common()):
            dictionary[word[0]] = i
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return dictionary, reversed_dictionary

    def word2vec(self, sentence: str):
        tokens = nltk.word_tokenize(sentence)
        v = [self.word_dict.get(token, 0) for token in tokens]
        return v

    def word_vecs(self):
        utterances, labels = self.read_json()
        # print(utterances)
        # print(self.label_dict)
        utterances = [self.word2vec(u) for u in utterances]
        labels = [self.label_dict[l] for l in labels]

        return utterances, labels

    def dataset(self, data_size: float, features=None):
        from collections import Counter
        import nltk

        utterances, labels = self.read_json()

        size = int(len(utterances) * data_size)
        utterances = utterances[:size]
        labels = labels[:size]

        if features is None:
            counter = Counter()
            for u in utterances:
                # print(u)
                tokens = nltk.word_tokenize(u)
                counter.update(tokens)

            features = [w for w, _ in counter.most_common()]

        vecs = []
        for u in utterances:
            tokens = nltk.word_tokenize(u)
            c = Counter(tokens)
            vec = tuple([c[t] for t in features])  # type: Tuple[float]
            vecs.append(vec)

        from dataset import Dataset

        return Dataset(tuple(features), tuple(labels), tuple(vecs))


if __name__ == '__main__':
    d = DSTC2.devset(100)
    u, l = d.word_vecs()
