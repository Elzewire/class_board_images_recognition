from recognition.data_loader import RuDataLoader
from recognition.main import FilePaths, train
from recognition.model import Model, DecoderType

if __name__ == '__main__':
    loader = RuDataLoader(FilePaths.fn_train_ru, Model.batchSize, Model.imgSize, Model.maxTextLen)
    # save characters of model for inference mode
    open(FilePaths.fn_char_list_ru, 'w', encoding='utf-8').write(str().join(loader.charList))

    # save words contained in dataset into file
    open(FilePaths.fn_corpus_ru, 'w', encoding='utf-8').write(str(' ').join(loader.trainWords + loader.validationWords))

    model = Model(loader.charList, decoder_type=DecoderType.BestPath, ru=True)
    train(model, loader)
