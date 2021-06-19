from difflib import get_close_matches
import numpy as np


def load_dict(lang='ru'):
    path = 'data/corpus.txt'
    if lang == 'ru':
        path = 'data/corpus_ru.txt'

    words = []

    f = open(path, encoding='utf-8')
    for line in f.readlines():
        for x in line.split(' '):
            words.append(x)

    return words


def dict_decode(sequences, lang='ru'):
    results = []
    for s in sequences:
        matches = set(get_close_matches(s, load_dict(lang), cutoff=.7))
        if matches:
            for m in matches:
                results.append(m)

    return results
