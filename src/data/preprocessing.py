"""https://www.kaggle.com/sudalairajkumar/getting-started-with-text-preprocessing"""
import pandas as pd
import string
from collections import Counter
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
from src.data.encoding_dict import EMOTICONS, UNICODE_EMO, CHAT_WORDS
from spellchecker import SpellChecker
from bs4 import BeautifulSoup
import torch
from src.utils import get_root_path
import os


class Preprocessor:
    def __init__(self):
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
        self.chat_words_list, self.chat_words_map_dict = CHAT_WORDS

    def save(self, stage: str):
        df = pd.read_csv(os.path.join(get_root_path(), "data", "raw", stage+".csv"))
        self._preprocess(df, stage)
        torch.save(self.df_out, os.path.join(get_root_path(), "data", "processed", stage+".pt"))

    def _preprocess(self, df: pd.DataFrame, stage: str):
        df_processed = pd.DataFrame()
        if stage == "train":
            df_processed["target"] = df["target"]
        df_processed["clean_text"] = df["text"].str.lower()
        df_processed["clean_text"] = df_processed["clean_text"].apply(lambda text: self.remove_punctuation(text))
        df_processed["clean_text"] = df_processed["clean_text"].apply(lambda text: self.remove_html(text))
        df_processed["clean_text"] = df_processed["clean_text"].apply(lambda text: self.remove_digits(text))
        df_processed["clean_text"] = df_processed["clean_text"].apply(lambda text: self.remove_urls(text))
        df_processed["clean_text"] = df_processed["clean_text"].apply(lambda text: self.remove_emoticons(text))
        df_processed["clean_text"] = df_processed["clean_text"].apply(lambda text: self.remove_emoji(text))
        df_processed["clean_text"] = df_processed["clean_text"].apply(lambda text: self.remove_special_characters(text))
        df_processed["clean_text"] = df_processed["clean_text"].apply(lambda text: self.remove_stopwords(text))
        cnt = Counter()
        for text in df_processed["clean_text"].values:
            for word in text.split():
                cnt[word] += 1
        freqwords = set([w for (w, wc) in cnt.most_common(10)])
        df_processed["clean_text"] = df_processed["clean_text"].apply(
            lambda text: self.remove_freqwords(text, freqwords))
        n_rare_words = 10
        rarewords = set([w for (w, wc) in cnt.most_common()[:-n_rare_words - 1:-1]])
        df_processed["clean_text"] = df_processed["clean_text"].apply(
            lambda text: self.remove_rarewords(text, rarewords))
        lemmatizer = WordNetLemmatizer()
        wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}
        df_processed["clean_text"] = df_processed["clean_text"].apply(
            lambda text: self.lem_words(text, lemmatizer, wordnet_map))
        self.df_out = df_processed[df_processed["clean_text"].map(len) > 0]

    def remove_punctuation(self, text: str):
        return text.translate(str.maketrans('', '', string.punctuation))

    def remove_stopwords(self, text: str):
        return " ".join([word for word in str(text).split() if word not in set(stopwords.words('english'))])

    def remove_freqwords(self, text: str, freqwords):
        return " ".join([word for word in str(text).split() if word not in freqwords])

    def remove_rarewords(self, text: str, rarewords):
        return " ".join([word for word in str(text).split() if word not in rarewords])

    def stem_words(self, text: str, stemmer):
        return " ".join([stemmer.stem(word) for word in text.split()])

    def lem_words(self, text: str, lemmatizer, wordnet_map: dict):
        pos_tagged_text = nltk.pos_tag(text.split())
        return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN))
                         for word, pos in pos_tagged_text])

    def remove_emoji(self, string):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', string)

    def remove_emoticons(self, text):
        emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')
        return emoticon_pattern.sub(r'', text)

    def convert_emoticons(self, text):
        for emot in EMOTICONS:
            text = re.sub(u'(' + emot + ')', "_".join(EMOTICONS[emot].replace(",", "").split()), text)
        return text

    def convert_emojis(self, text):
        for emot in UNICODE_EMO:
            text = re.sub(r'(' + emot + ')', "_".join(UNICODE_EMO[emot].replace(",", "").replace(":", "").split()),
                          text)
        return text

    def remove_urls(self, text):
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)

    def remove_html(self, text):
        html_pattern = re.compile('<.*?>')
        return html_pattern.sub(r'', text)

    def remove_html_bs4(self, text):
        return BeautifulSoup(text, "lxml").text

    def chat_words_conversion(self, text):
        new_text = []
        for w in text.split():
            if w.upper() in self.chat_words_list:
                new_text.append(self.chat_words_map_dict[w.upper()])
            else:
                new_text.append(w)
        return " ".join(new_text)

    def remove_special_characters(self, text):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())

    def remove_digits(self, text):
        return ' '.join(re.sub("\d+", " ", text).split())


if __name__ == "__main__":
    pre = Preprocessor()
    pre.save("train")
