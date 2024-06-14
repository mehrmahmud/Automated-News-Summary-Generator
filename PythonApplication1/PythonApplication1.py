import newspaper
from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import nltk

LANGUAGE = "english"
SENTENCES_COUNT = 5

def fetch_news_articles(url):
    article = newspaper.Article(url)
    article.download()
    article.parse()
    return article.text

def generate_summary(text):
    parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)

    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)

    summary = ""
    for sentence in summarizer(parser.document, SENTENCES_COUNT):
        summary += str(sentence) + " "

    return summary

def main():
    url = input("Enter the URL of the news article: ")
    text = fetch_news_articles(url)
    summary = generate_summary(text)
    print("Summary:")
    print(summary)

if __name__ == "__main__":
    main()
