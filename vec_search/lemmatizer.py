import nltk
from pymystem3 import Mystem
from nltk.stem import WordNetLemmatizer

def lemmatize(text):
    # Заменить символы пробелами
    text = text.replace('_', ' ')

    # Разбить текст на слова
    tokenizer = nltk.RegexpTokenizer(r"[А-Яа-я]+")
    ru_words = tokenizer.tokenize(text)

    # Лемматизация
    m = Mystem()

    ru_lemmas = m.lemmatize(" ".join(ru_words))
    l2 = []
    for l in ru_lemmas:
        if l != ' ' and l != '\n':
            l2.append(l)

    tokenizer = nltk.RegexpTokenizer(r"[A-Za-z]+")
    en_words = tokenizer.tokenize(text)

    lemmatizer = WordNetLemmatizer()
    en_lemmas = [lemmatizer.lemmatize(w) for w in en_words]

    for l in en_lemmas:
        l2.append(l.lower())

    return l2