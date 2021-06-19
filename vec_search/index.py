import os
from math import log2

from vec_search.lemmatizer import lemmatize
from vec_search.loader import load_indexed


def index_documents():
    # Индексирование РПД
    dir = '../data/РПД/'
    index = '../data/indexes/index.csv'
    ix = 0
    file = open(index, 'w')
    for f in os.listdir(dir):
        if f.split('.')[1] == 'txt':
            file.write('{};{};{}\n'.format(f, ix, f.split('.')[0]))  # имя файла, индекс, название
            ix += 1
    file.close()


def generate_inv_index():
    # Построение инвертированных индексов
    docs = load_indexed()
    dir = '../data/РПД/'
    rev_index = {}
    for d in docs:
        file = open(os.path.join(dir, d[0]), 'r', encoding='utf-8')
        lemmas = lemmatize(file.read())
        file.close()
        for l in lemmas:
            if l not in rev_index.keys():
                rev_index[l] = set()
            rev_index[l].add(d[1])

    # Сохранение индексов
    inv_index = '../data/indexes/inv_index.csv'
    file = open(inv_index, 'w')
    for k in rev_index.keys():
        file.write('{} {}\n'.format(k, ' '.join(str(s) for s in rev_index[k])))
    file.close()


def generate_tf_idf_index():
    # Рассчёт IDF по инвертированному индексу
    docs = load_indexed()
    idf = {}
    inv_index = '../data/indexes/inv_index.csv'
    file = open(inv_index, 'r')
    for l in file.readlines():
        lemma = l.split(' ')[0]
        idf[lemma] = log2(len(docs) / len(l.split(' ')[1:]))
    file.close()

    # Предзагрузка лемматизированных документов
    dir = '../data/РПД/'
    doc_lemmas = []
    for d in docs:
        file = open(os.path.join(dir, d[0]), 'r', encoding='utf-8')
        lemmas = lemmatize(file.read())
        doc_lemmas.append(lemmas)

    # Рассчёт TF
    tf = {}
    for k in idf.keys():
        entries = []
        for lemmas in doc_lemmas:
            i = 0
            for l in lemmas:
                if k == l:
                    i += 1
            entries.append(i / len(lemmas))
        tf[k] = entries

    # Рассчёт TF-IDF векторов
    tf_idf = {}
    for k in idf.keys():
        tf_idf[k] = [x * idf[k] for x in tf[k]]

    # Сохранение индекса в файл
    tf_idf_index = '../data/indexes/tf_idf_index.csv'
    file = open(tf_idf_index, 'w')
    for k in tf_idf.keys():
        file.write('{} {} {}\n'.format(k, idf[k], ' '.join(str(s) for s in tf_idf[k])))
    file.close()


if __name__ == '__main__':
    index_documents()
    generate_inv_index()
    generate_tf_idf_index()
