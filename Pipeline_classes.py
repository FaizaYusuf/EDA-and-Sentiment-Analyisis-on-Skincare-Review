import spacy
from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import numpy as np
import re
from sklearn import set_config
from spacytextblob.spacytextblob import SpacyTextBlob

set_config("diagram")
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("spacytextblob")

class ConvertToLower(BaseEstimator, TransformerMixin):
    """_This class convert the review test lower case
    """
    def __init__(self) -> None:
        """ this contains parameters to pass"""
        pass

    def fit(self, X, y=None) -> None:
        """This is a required step to work with sklearn Pipelines."""
        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        """This applies the transformation."""
        X = X.copy()
        column = "Review_Text"
        X[column] = X[column].apply(lambda x: str(x).lower())
        return X


class RemoveStopwords(BaseEstimator, TransformerMixin):
    """this class remove stopwords in the review column
    """
    def __init__(self) -> None:
        """ this contains parameters to pass"""
        self.nlp = spacy.load("en_core_web_sm")
        self.stopwords = self.nlp.Defaults.stop_words

    def remove_stop_words(self, *, text: str):
        """The function accepts accepts string and apply stowards to each words in the string"""
        doc = self.nlp(text)
        new_doc = ""
        for token in doc:
            if (token.text).lower() not in self.stopwords:
                new_doc = new_doc + " " + str(token.text)
        return new_doc

    def fit(self, X, y=None) -> None:
        """This is a required step to work with sklearn Pipelines."""
        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        """This applies the transformation."""
        X = X.copy()
        column = "Review_Text"
        X[column] = X[column].apply(lambda x: self.remove_stop_words(text=x))
        return X


class RemoveSpecicialCharacter(BaseEstimator, TransformerMixin):
    """This class removes special keywords in the review column
    """
    def __init__(self) -> None:
        """ this contains parameters to pass"""
        pass

    def fit(self, X, y=None) -> None:
        """This is a required step to work with sklearn Pipelines."""
        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        """This applies the transformation."""
        pattern = r"[^\w\s]"
        replace = ""
        X = X.copy()
        column = "Review_Text"
        X[column] = X[column].apply(
            lambda x: re.sub(pattern=pattern, repl=replace, string=x)
        )
        return X


class Lemmertize(BaseEstimator, TransformerMixin):
    """ this class lemmertize each words in the review text"""
    def __init__(self) -> None:
        """ this contains parameters to pass"""
        pass

    def lemma(self, *, text: str):
        """ this function accept sting and lemmertize each word in the string"""
        doc = nlp(text)
        new_doc = ""
        for token in doc:
            new_doc = new_doc + " " + token.lemma_
        return new_doc

    def fit(self, X, y=None) -> None:
        """This is a required step to work with sklearn Pipelines."""
        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        """This applies the transformation."""
        X = X.copy()
        column = "Review_Text"
        X[column] = X[column].apply(lambda x: self.lemma(text=x))
        return X


class CreateSentimentScore(BaseEstimator, TransformerMixin):
    """ this function create new column store the sentiment score of each review """
    def __init__(self) -> None:
        """ this contains parameters to pass"""
        pass

    def fit(self, X, y=None) -> None:
        """This is a required step to work with sklearn Pipelines."""
        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        """This applies the transformation."""
        X = X.copy()
        column = "Review_Text"
        new_col = "Sentiment"
        X[new_col] = X[column].apply(lambda x: round(nlp(x)._.polarity, 2))
        return X


class CreateRatingThree(BaseEstimator, TransformerMixin):
    """ this class create creates a new column and store the rating of sentiment score to either 
        positive, negative or nuetral
    """
    def __init__(self) -> None:
        """ this contains parameters to pass"""
        pass

    def fit(self, X, y=None) -> None:
        """This is a required step to work with sklearn Pipelines."""
        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        """This applies the transformation."""
        X = X.copy()
        column = "Sentiment"
        new_col = "Rating(Three)"
        X[new_col] = X[column].apply(
            lambda x: "negative" if x < 0 else ("nuetral" if x == 0 else "positive")
        )
        return X


class CreateRatingTwo(BaseEstimator, TransformerMixin):
    """ this class create creates a new column and store the rating of sentiment score to either 
        1 or 0
    """
    def __init__(self) -> None:
        """ this contains parameters to pass"""
        pass

    def fit(self, X, y=None) -> None:
        """This is a required step to work with sklearn Pipelines."""
        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        """This applies the transformation."""
        X = X.copy()
        column = "Sentiment"
        new_col = "Rating"
        X[new_col] = X[column].apply(lambda x: 1 if x >= 0 else 0)
        return X


class CreateSkinIssue(BaseEstimator, TransformerMixin):
    """ this class create creates a new column and store the skincare issues to either 
        yes or not sure
    """
    def __init__(self, *, pattern) -> None:
        """ this contains parameters to pass"""
        self.pattern = pattern

    def fit(self, X, y=None) -> None:
        """This is a required step to work with sklearn Pipelines."""
        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        """This applies the transformation."""
        X = X.copy()
        column = "Review_Text"
        new_col = "skincare_issue"
        X[new_col] = np.where(
            X[column].str.strip().str.contains(pat=self.pattern, flags=re.I),
            "yes",
            "not really",
        )
        return X