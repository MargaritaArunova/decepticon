DATASET_PREFIX = 'dataset/'
SUFFIX_FILE_EN = '.de-en.en'
SUFFIX_FILE_DE = '.de-en.de'


def import_train_texts():
    source_texts, target_texts = [], []
    with open(DATASET_PREFIX + 'train' + SUFFIX_FILE_DE, encoding='utf8') as file:
        source_texts = file.readlines()
    with open(DATASET_PREFIX + 'train' + SUFFIX_FILE_EN, encoding='utf8') as file:
        target_texts = file.readlines()
    return source_texts, target_texts


def import_validation_texts():
    source_texts, target_texts = [], []
    with open(DATASET_PREFIX + 'val' + SUFFIX_FILE_DE, encoding='utf8') as file:
        source_texts = file.readlines()
    with open(DATASET_PREFIX + 'val' + SUFFIX_FILE_EN, encoding='utf8') as file:
        target_texts = file.readlines()
    return source_texts, target_texts


def save_validation_prediction(predicted_txts, target_txts):
    with open('pred.txt', 'w') as f:
        for line in predicted_txts:
            f.write(f"{line}\n")
    with open('target.txt', 'w') as f:
        for line in target_txts:
            f.write(f"{line}")


def make_prediction(model):
    source_texts, predicted_texts = [], []
    with open(DATASET_PREFIX + 'test1' + SUFFIX_FILE_DE, encoding='utf8') as file:
        source_texts = file.readlines()

    for source in source_texts:
        prediction = model.inference(source)
        predicted_texts.append(prediction)

    with open('pred_test.txt', 'w') as f:
        for line in predicted_texts:
            f.write(f"{line}\n")

    return predicted_texts
