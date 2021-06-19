def load_indexed():
    # Загрузка индексированных документов
    docs = []
    index = 'data/indexes/index.csv'
    file = open(index, 'r')
    for l in file.readlines():
        line = l.split(';')
        docs.append([line[0], line[1], line[2]])
    file.close()

    return docs