import json
from collections import Counter
from enum import Enum
from typing import Tuple

import nltk


class DSTC2:
    """
    Class wrapping the DSTC2 dataset
    """

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
        """
        Initialize the class
        :param mode: which dataset to get.
        :param n_word: how many words to keep in the vocabulary, the rest will be <unk>
        """
        self.mode = mode
        # the json files
        self.label_json, self.log_json = list(self._filepath('label')), list(self._filepath('log'))
        # the data as word dictionary, int encoded
        self.word_dict = self._build_set(n_word)
        # the labels as label dictionary, int encoded
        self.label_dict = self._build_label()

    def _filepath(self, which_one: str):
        """
        method for getting the json file.
        :param which_one: either 'test' or 'traindev'
        :return:
        """
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
        """
        read the dataset json file. Gets only the utterance and the first label for that utterance
        :return:
        """
        utterances, labels = [], []
        for log in self.log_json:
            for turn in log['turns']:
                utterance = turn['output']['transcript']
                label = turn['output']['dialog-acts'][0]['act']
                utterances.append(utterance)
                labels.append(label)

        return utterances, labels

    def _build_set(self, n_words):
        """
        Build the word dictionary from the dataset.
        :param n_words: how many words to keep in the vocabulary
        :return:
        """
        # count all words
        counter = Counter()
        utterances, labels = self.read_json()
        for utterance in utterances:
            tokens = nltk.word_tokenize(utterance)
            counter.update(tokens)

        # generate an int representation
        count = [['UNK', -1]]
        count.extend(counter.most_common(n_words - 1))

        # convert the int representation into a dictionary
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
        return dictionary

    def _build_label(self):
        """
        encode labels into int representation
        :return:
        """
        counter = Counter()
        _, labels = self.read_json()
        counter.update(labels)
        dictionary = dict()
        for i, word in enumerate(counter.most_common()):
            dictionary[word[0]] = i
        return dictionary

    def word2vec(self, sentence: str):
        """
        method for converting a sentence into List[int] based on the encoded word dictionary
        :param sentence:
        :return:
        """
        tokens = nltk.word_tokenize(sentence)
        v = [self.word_dict.get(token, 0) for token in tokens]
        return v

    def word_vecs(self, raw_label=False):
        """
        Function returning the feature vectors, labels as int representations
        :param raw_label: should the function return the labels encoded or not
        :return:
        """
        utterances, labels = self.read_json()
        # print(utterances)
        # print(self.label_dict)
        utterances = [self.word2vec(u) for u in utterances]
        if raw_label:
            labels = labels
        else:
            labels = [self.label_dict[l] for l in labels]

        return utterances, labels

    def dataset(self, data_size: float = 1.0, features=None):
        """
        Function returning the DSTC2 data as a Dataset class representation
        :param data_size: percentage of slicing the data
        :param features: if there's already a word vocabulary, provide it here
        :return:
        """
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
