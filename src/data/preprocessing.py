# https://www.kaggle.com/sudalairajkumar/getting-started-with-text-preprocessing
# https://towardsdatascience.com/cleaning-preprocessing-text-data-by-building-nlp-pipeline-853148add68a
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
from src.util import get_root_path
import os
from typing import Optional, List
import wordsegment
import spacy


class Preprocessor:
    def __init__(self):
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
        wordsegment.load()
        self.chat_words_list, self.chat_words_map_dict = CHAT_WORDS

    def run(self, stage: str, features: Optional[List[str]] = None):
        df = pd.read_csv(os.path.join(get_root_path(), "data", "raw", stage+".csv"))
        if stage == "train":
            self._preprocess(df, features)
        elif stage == "test":
            self._preprocess_test(df, features)
        else:
            return
        print(f"Saving dataframe to {os.path.join(get_root_path(), 'data', 'processed', stage+'.pt')}")
        torch.save(self.df_out, os.path.join(get_root_path(), "data", "processed", stage+".pt"))

    def _preprocess(self, df: pd.DataFrame, features: Optional[List[str]] = None) -> None:
        df_processed = pd.DataFrame()
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
            lambda text: self.remove_freqwords(text, freqwords)
        )
        n_rare_words = 10
        rarewords = set([w for (w, wc) in cnt.most_common()[:-n_rare_words - 1:-1]])
        df_processed["clean_text"] = df_processed["clean_text"].apply(
            lambda text: self.remove_rarewords(text, rarewords)
        )
        lemmatizer = WordNetLemmatizer()
        wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}
        df_processed["clean_text"] = df_processed["clean_text"].apply(
            lambda text: self.lem_words(text, lemmatizer, wordnet_map)
        )
        self.df_out = df_processed[df_processed["clean_text"].map(len) > 0]
        # add additional features
        if features:
            for feature in features:
                print(f"Adding feature: {feature}")
                self._add_feature(df, feature)
                if feature == "location":
                    self.nlp = spacy.load("en_core_web_sm")
                    self.df_out["location"] = self.df_out["location"].apply(
                        lambda text: self.filter_ner(text, ["GPE", "LOC"])
                    )

    def _preprocess_test(self, df: pd.DataFrame, features: Optional[List[str]] = None) -> None:
        df_processed = pd.DataFrame()
        df_processed["clean_text"] = df["text"].str.lower()
        df_processed["clean_text"] = df_processed["clean_text"].apply(lambda text: self.remove_punctuation(text))
        df_processed["clean_text"] = df_processed["clean_text"].apply(lambda text: self.remove_html(text))
        df_processed["clean_text"] = df_processed["clean_text"].apply(lambda text: self.remove_digits(text))
        df_processed["clean_text"] = df_processed["clean_text"].apply(lambda text: self.remove_urls(text))
        df_processed["clean_text"] = df_processed["clean_text"].apply(lambda text: self.remove_emoticons(text))
        df_processed["clean_text"] = df_processed["clean_text"].apply(lambda text: self.remove_emoji(text))
        df_processed["clean_text"] = df_processed["clean_text"].apply(lambda text: self.remove_special_characters(text))
        lemmatizer = WordNetLemmatizer()
        wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}
        df_processed["clean_text"] = df_processed["clean_text"].apply(
            lambda text: self.lem_words(text, lemmatizer, wordnet_map))
        self.df_out = df_processed
        if features:
            for feature in features:
                print(f"Adding feature: {feature}")
                self._add_feature(df, feature)
                if feature == "location":
                    self.nlp = spacy.load("en_core_web_sm")
                    self.df_out["location"] = self.df_out["location"].apply(
                        lambda text: self.filter_ner(text, ["GPE", "LOC"])
                    )
        self.df_out["id"] = df["id"]  # for submission to Kaggle competition

    def _add_feature(self, df: pd.DataFrame, feature: str) -> None:
        df[feature].fillna("NAN", inplace=True)
        df[feature] = df[feature].map(lambda x: self.remove_punctuation(x))
        df[feature] = df[feature].map(lambda x: self.word_segmentation(x))
        self.df_out[feature] = df[feature]

    def word_segmentation(self, text: str) -> str:
        return " ".join(wordsegment.segment(text))

    def run_ner(self, text) -> List[str]:
        doc = self.nlp(text)
        return [X.label_ for X in doc.ents]

    def filter_ner(self, text, target_entities: List[str]) -> str:
        doc = self.nlp(text)
        locations = [X.lemma_ for X in doc.ents if X.label_ in target_entities]
        if len(locations) == 0:
            return "nan"
        else:
            return " ".join(locations)

    def remove_punctuation(self, text: str) -> str:
        return text.translate(str.maketrans('', '', string.punctuation))

    def remove_stopwords(self, text: str) -> str:
        return " ".join([word for word in str(text).split() if word not in set(stopwords.words('english'))])

    def remove_freqwords(self, text: str, freqwords) -> str:
        return " ".join([word for word in str(text).split() if word not in freqwords])

    def remove_rarewords(self, text: str, rarewords) -> str:
        return " ".join([word for word in str(text).split() if word not in rarewords])

    def stem_words(self, text: str, stemmer) -> str:
        return " ".join([stemmer.stem(word) for word in text.split()])

    def lem_words(self, text: str, lemmatizer, wordnet_map: dict) -> str:
        pos_tagged_text = nltk.pos_tag(text.split())
        return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN))
                         for word, pos in pos_tagged_text])

    def remove_emoji(self, string) -> str:
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', string)

    def remove_emoticons(self, text) -> str:
        emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')
        return emoticon_pattern.sub(r'', text)

    def convert_emoticons(self, text) -> str:
        for emot in EMOTICONS:
            text = re.sub(u'(' + emot + ')', "_".join(EMOTICONS[emot].replace(",", "").split()), text)
        return text

    def convert_emojis(self, text) -> str:
        for emot in UNICODE_EMO:
            text = re.sub(r'(' + emot + ')', "_".join(UNICODE_EMO[emot].replace(",", "").replace(":", "").split()),
                          text)
        return text

    def remove_urls(self, text) -> str:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)

    def remove_html(self, text) -> str:
        html_pattern = re.compile('<.*?>')
        return html_pattern.sub(r'', text)

    def remove_html_bs4(self, text) -> str:
        return BeautifulSoup(text, "lxml").text

    def chat_words_conversion(self, text) -> str:
        new_text = []
        for w in text.split():
            if w.upper() in self.chat_words_list:
                new_text.append(self.chat_words_map_dict[w.upper()])
            else:
                new_text.append(w)
        return " ".join(new_text)

    def remove_special_characters(self, text) -> str:
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())

    def remove_digits(self, text) -> str:
        return ' '.join(re.sub("\d+", " ", text).split())


if __name__ == "__main__":
    pre = Preprocessor()
    pre.run("test", features=["location", "keyword"])
