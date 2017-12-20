import logging
import telebot
import pandas as pd
import numpy as np
from model_def import deepmoji_architecture, load_specific_weights
from global_variables import NB_TOKENS
from sentence_tokenizer import SentenceTokenizer
from start_train import load_vocab
from data_tweets.convert import emoji_dict as emoji_dict_

BOT_TOKEN = '449023090:AAGGUHwOCHcPVxxEsHWzoxLV1cjcNByp96E'
LOG_FILE = 'bot_log.log'
MODEL_FILE = "..\\data_tweets\\model.hdf5"
DATASET_PATH_PRETRAINED = '..\\data_tweets\\twitter_10_classes.csv'


class TestBot:
    def __init__(self, emoji_model, emoji_dict):
        # Bot initialization
        self.bot = telebot.TeleBot(BOT_TOKEN)

        # Logger initialization
        self.logger = logging.getLogger('BotLogger')
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(LOG_FILE)

        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.info('NLP init')
        self.emoji_model = emoji_model
        self.emoji_model.predict(np.ones((1, 200)))
        self.emoji_dict_inverse = {v: k for k, v in emoji_dict.items()}
        vocab = build_vocab()
        self.st = SentenceTokenizer(vocab, 200)
        self.logger.info('Bot init done')

        @self.bot.message_handler(commands=['start', 'help'])
        def start(message):
            self.bot.send_message(chat_id=message.chat.id, text='TODO')

        @self.bot.message_handler(func=lambda message: True, content_types=['text'])
        def parse_message(message):
            tokenized_text = self.st.tokenize_sentences([message.text])[0]
            best_emoji = self.emoji_dict_inverse[np.argmax(self.emoji_model.predict(tokenized_text)[0])]
            self.bot.send_message(chat_id=message.chat.id, text=str(best_emoji))

    def __del__(self):
        self.bot.stop_polling()
        self.logger.info('Bot deinit done')

    def start(self):
        self.bot.polling(none_stop=True)


def build_vocab(data_path=DATASET_PATH_PRETRAINED):
    vocab = {
        "CUSTOM_MASK": 0,
        "CUSTOM_UNKNOWN": 1,
        "CUSTOM_AT": 2,
        "CUSTOM_URL": 3,
        "CUSTOM_NUMBER": 4,
        "CUSTOM_BREAK": 5,
        "CUSTOM_BLANK_6": 6,
        "CUSTOM_BLANK_7": 7,
        "CUSTOM_BLANK_8": 8,
        "CUSTOM_BLANK_9": 9
    }
    data = pd.read_csv(data_path, sep='\t')
    data = data.dropna()
    data.to_csv(data_path, sep='\t', index=False)
    vocab = load_vocab(data, vocab)
    return vocab


def load_emoji_model(model_file):
    model = deepmoji_architecture(nb_classes=10, nb_tokens=NB_TOKENS, maxlen=200)
    # model.load_weights(model_file)
    load_specific_weights(model, model_file)
    return model


if __name__ == '__main__':

    emoji_model_ = load_emoji_model(MODEL_FILE)
    bot = TestBot(emoji_model_, emoji_dict_)
    bot.start()
