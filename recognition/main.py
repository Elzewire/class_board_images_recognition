from __future__ import division
from __future__ import print_function

import argparse
import os

import cv2
import editdistance
import tensorflow as tf

from recognition.data_loader import Batch, DataLoader
from recognition.model import Model, DecoderType
from recognition.sample_preprocessor import preprocess


class FilePaths:
    """filenames and paths to data"""
    fn_char_list = 'model/charList.txt'
    fn_accuracy = 'model/accuracy.txt'
    fn_train = 'data/'
    fn_infer = 'data/test.png'
    fn_corpus = 'data/corpus.txt'

    fn_train_ru = 'data/words/'
    fn_char_list_ru = 'model_ru/charList.txt'
    fn_accuracy_ru = 'model_ru/accuracy.txt'
    fn_corpus_ru = 'data/corpus_ru.txt'


def train(model, loader):
    """train NN"""
    epoch = 0  # number of training epochs since start
    best_char_error_rate = float('inf')  # best valdiation character error rate
    no_improvement_since = 0  # number of epochs no improvement of character error rate occured
    early_stopping = 5  # stop training after this number of epochs without improvement
    while True:
        epoch += 1
        print('Epoch:', epoch)

        # train
        print('Train NN')
        loader.train_set()
        while loader.has_next():
            iterInfo = loader.get_iterator_info()
            batch = loader.get_next()
            loss = model.train_batch(batch)
            print('Batch:', iterInfo[0], '/', iterInfo[1], 'Loss:', loss)

        # validate
        char_error_rate = validate(model, loader)

        # if best validation accuracy so far, save model parameters
        if char_error_rate < best_char_error_rate:
            print('Character error rate improved, save model')
            best_char_error_rate = char_error_rate
            no_improvement_since = 0
            model.save()
            open(FilePaths.fn_accuracy, 'w').write(
                'Validation character error rate of saved model: %f%%' % (char_error_rate * 100.0))
        else:
            print('Character error rate not improved')
            no_improvement_since += 1

        # stop training if no more improvement in the last x epochs
        if no_improvement_since >= early_stopping:
            print('No more improvement since %d epochs. Training stopped.' % early_stopping)
            break


def validate(model, loader):
    """validate NN"""
    print('Validate NN')
    loader.validation_set()
    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0
    numWordTotal = 0
    while loader.has_next():
        iterInfo = loader.get_iterator_info()
        print('Batch:', iterInfo[0], '/', iterInfo[1])
        batch = loader.get_next()
        (recognized, _) = model.infer_batch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
            numWordTotal += 1
            dist = editdistance.eval(recognized[i], batch.gtTexts[i])
            numCharErr += dist
            numCharTotal += len(batch.gtTexts[i])
            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gtTexts[i] + '"', '->',
                  '"' + recognized[i] + '"')

    # print validation result
    charErrorRate = numCharErr / numCharTotal
    wordAccuracy = numWordOK / numWordTotal
    print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate * 100.0, wordAccuracy * 100.0))
    return charErrorRate


def infer(model, fn_img):
    """recognize text in image provided by file path"""
    img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
    img = preprocess(img, Model.imgSize)
    batch = Batch(None, [img])
    (recognized, probability) = model.infer_batch(batch, True)
    print('Recognized:', '"' + recognized[0] + '"')
    print('Probability:', probability[0])


def predict(images):
    """recognize text in image provided by file path"""
    model = Model(open(FilePaths.fn_char_list).read(), DecoderType.BestPath, must_restore=True)
    results = []
    k = 0
    for x in images:
        tf.reset_default_graph()
        img = preprocess(x, Model.imgSize)
        k += 1
        batch = Batch(None, [img])
        (recognized, probability) = model.infer_batch(batch)
        results.append(recognized[0])
    return results


def predict_ru(images):
    """recognize text in image provided by file path"""
    model = Model(open(FilePaths.fn_char_list_ru, encoding='utf-8').read(), DecoderType.BestPath, must_restore=True, ru=True)
    results = []
    k = 0
    for x in images:
        tf.reset_default_graph()
        img = preprocess(x, Model.imgSize)
        k += 1
        batch = Batch(None, [img])
        (recognized, probability) = model.infer_batch(batch)
        results.append(recognized[0])
    return results


def main():
    """main function"""
    # optional command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='train the NN', action='store_true')
    parser.add_argument('--validate', help='validate the NN', action='store_true')
    parser.add_argument('--beamsearch', help='use beam search instead of best path decoding', action='store_true')
    parser.add_argument('--wordbeamsearch', help='use word beam search instead of best path decoding',
                        action='store_true')
    parser.add_argument('--dump', help='dump output of NN to CSV file(s)', action='store_true')

    args = parser.parse_args()

    decoderType = DecoderType.BestPath
    if args.beamsearch:
        decoderType = DecoderType.BeamSearch
    elif args.wordbeamsearch:
        decoderType = DecoderType.WordBeamSearch

    # train or validate on IAM dataset
    if args.train or args.validate:
        # load training data, create TF model
        loader = DataLoader(FilePaths.fn_train, Model.batchSize, Model.imgSize, Model.maxTextLen)

        # save characters of model for inference mode
        open(FilePaths.fn_char_list, 'w').write(str().join(loader.charList))

        # save words contained in dataset into file
        open(FilePaths.fn_corpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))

        # execute training or validation
        if args.train:
            model = Model(loader.charList, decoderType)
            train(model, loader)
        elif args.validate:
            model = Model(loader.charList, decoderType, must_restore=True)
            validate(model, loader)

    # infer text on test image
    else:
        print(open(FilePaths.fn_accuracy).read())
        model = Model(open(FilePaths.fn_char_list).read(), decoderType, must_restore=True, dump=args.dump)
        infer(model, FilePaths.fn_infer)
