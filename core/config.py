IMAGE_PATH = './data/imagenesStacked'
DEVICE = "cuda:0"

SPANISH_BERT = 'dccuchile/bert-base-spanish-wwm-uncased'
DEFAULT_BERT = 'bert-base-uncased'

ARG_BERT = {
    "learning_rate":  2.2777082441093773e-05, # #
    "weight_decay":  0.004622139246632094, #
}

ARG_FILES_SP = {
    "text_field": "descripcion",
    "label_field": "tipo",
    "source_field": "fuente",
    "image_field": "imagen"
}

ARG_FILES_EN = {
    "text_field": "description",
    "label_field": "type",
    "source_field": "source",
    "image_field": "image"
}

TRAIN = './data/english_data/train.csv'
TEST = './data/english_data/test.csv'
VALID = './data/english_data/val.csv'

# En ingles
TRAIN_CSV_EN = './data/english_data/train.csv'
TEST_CSV_EN = './data/english_data/test.csv'
VAL_CSV_EN = './data/english_data/val.csv'

ARG_RESNET =  {
    "learning_rate":  0.0002, 
    "weight_decay": 0.01, 
}

ARG_DENSENET = {
    'learning_rate': 0.01,
    "weight_decay": 0.001
}

ARG_ALEXNET = {
    'learning_rate': 0.00005,
    "weight_decay": 0.02,
}

ARG_CONVNEXT = {
    'learning_rate': 0.001,
    "weight_decay": 1e-8,
}

ARG_TRANFORMER = {
    'learning_rate': 0.001,
    "weight_decay": 0.01,
}

NUMBER_EPOCHS = 10000
BATCH_SIZE = 24

LANGUAGE_CHOICES = ['sp','en']

OUTPUT_FOLDER_EN = { 'Bert': './output_english/Bert/',
                 'Beto': './output_english/Beto/',
                 'ResNet18': './output_english/ResNet18/',
                 'ResBet18': './output_english/ResBet18/',
                 'ResBet50': './output_english/ResBet50/',
                 'ResNet50': './output_english/ResNet50/',
                 'DenseNet121': './output_english/DenseNet121/',
                 'DenseNet169': './output_english/DenseNet169/',
                 'DenseBet121': './output_english/DenseBet121/',
                 'DenseBet169': './output_english/DenseBet169/',
                 'AlexNet': './output_english/AlexNet/',
                 'AlexBet': './output_english/AlexBet/',
                 'ConvNeXt_Base': './output_english/ConvNeXt_Base/',
                 'ConvNeXt_Small': './output_english/ConvNeXt_Small/',
                 'Tranformer': './output_english/Transformer/',
                 'ConvBet_Small': './output_english/ConvBet_Small/',
                 'ConvBet_Base': './output_english/ConvBet_Base/',
                 'T16': './output_english/T16/',
                 'T16Bet': './output_english/T16Bet/'}

OUTPUT_FOLDER_SP = { 'Bert': './output/Bert/',
                 'Beto': './output/Beto/',
                 'ResNet18': './output/ResNet18/',
                 'ResBet18': './output/ResBet18/',
                 'ResBet50': './output/ResBet50/',
                 'ResNet50': './output/ResNet50/',
                 'DenseNet121': './output/DenseNet121/',
                 'DenseNet169': './output/DenseNet169/',
                 'DenseBet121': './output/DenseBet121/',
                 'DenseBet169': './output/DenseBet169/',
                 'AlexNet': './output/AlexNet/',
                 'AlexBet': './output/AlexBet/',
                 'ConvNeXt_Base': './output/ConvNeXt_Base/',
                 'ConvNeXt_Small': './output/ConvNeXt_Small/',
                 'Tranformer': './output/Transformer/',
                 'ConvBet_Small': './output/ConvBet_Small/',
                 'ConvBet_Base': './output/ConvBet_Base/',
                 'T16': './output/T16/',
                 'T16Bet': './output/T16Bet/'}