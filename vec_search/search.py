from math import sqrt

from vec_search.lemmatizer import lemmatize

# Косинусное сходство
from vec_search.loader import load_indexed


def cos_sim(v1, v2):
    len_v1 = sqrt(sum([x * x for x in v1]))
    len_v2 = sqrt(sum([x * x for x in v2]))

    mult = 0
    for x, y in zip(v1, v2):
        mult += x * y

    return mult / (len_v1 * len_v2) if len_v1 * len_v2 != 0 else 0


def decode_output(results):
    decoded = []
    docs = load_indexed()
    for r in results:
        if r[1] > 0:
            decoded.append((r[1], docs[r[0]][2], '{}.pdf'.format(docs[r[0]][2])))
    decoded.sort(reverse=True)
    return decoded


def vector_search(q):
    q_lemmas = lemmatize(q)
    docs = load_indexed()

    # Загрузка TF-IDF индексов из файла
    tf_idf = {}
    idf = {}
    tf_idf_index = 'data/indexes/tf_idf_index.csv'
    file = open(tf_idf_index, 'r')
    for l in file.readlines():
        lemma = l.split(' ')[0]
        idf[lemma] = float(l.split(' ')[1])
        tf_idf[lemma] = [float(x) for x in l.split(' ')[2:]]
    file.close()

    # Вычисление вектора запроса
    ql_counts = {}
    for ql in q_lemmas:
        if ql in ql_counts.keys():
            ql_counts[ql] += 1
        else:
            ql_counts[ql] = 1

    max_count = 0
    for k in ql_counts.keys():
        if ql_counts[k] > max_count:
            max_count = ql_counts[k]

    q_vector = []
    for k in idf.keys():
        if k in ql_counts.keys():
            q_vector.append((ql_counts[k] / max_count) * idf[k])
        else:
            q_vector.append(0)

    # Рассчёт длин векторов
    vectors = []
    for i in range(len(docs)):
        v = []
        for k in tf_idf.keys():
            v.append(tf_idf[k][i])
        vectors.append(v)

    results = []
    for i, v in enumerate(vectors):
        results.append((i, cos_sim(v, q_vector)))

    results.sort(reverse=True, key=lambda tup: tup[1])
    doc_indexes = []
    for i, x in results:
        if x > 0:
            doc_indexes.append((i, x))

    return decode_output(results)


if __name__ == '__main__':
    print(vector_search('Перестановки'))
