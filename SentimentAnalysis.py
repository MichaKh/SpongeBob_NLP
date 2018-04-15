import os
import re
import string
from operator import itemgetter
import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity, stopwords
from nltk.sentiment import SentimentAnalyzer, SentimentIntensityAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def read_files_to_dict(dir_path):
    """
    Read file in specified path
    :param dir_path: Path of file to read
    :return:
    """
    conversations = {}
    for filename in os.listdir(dir_path):
        if filename.endswith(".txt"):
            file_path = os.path.abspath(os.path.join(dir_path, filename))
            with open(file_path, 'r') as f:
                x = f.readlines()
                content_unicode = unicode(x[0], encoding='utf-8', errors='replace')
                sentences = content_unicode.split('@')
                if not filename in conversations:
                    conversations[filename.split('.txt')[0]] = sentences
    return conversations


def preprocess(doc):
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    digits = '0123456789'

    tokenized_text = nltk.word_tokenize(doc.lower())
    cleaned_text = [t for t in tokenized_text if t not in stop and re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', t)]
    cleaned_text = ' '.join(w.strip('.\\?@\'~{}()') for w in cleaned_text)
    return cleaned_text


def get_all_character_conversations(character, conversations):
    char_conversations = {}

    if character in conversations:
        item_pairs = [(key, value) for key, value in conversations.items() if character+"-" in key]
        for item in item_pairs:
            char_conversations[item[0]] = item[1]
    return char_conversations


def classify_conversation(conversation, analyzer):
    conversation_sentiment = 0.0
    for sentence in conversation:
        compound_score, label = classify(sentence, analyzer)
        conversation_sentiment += compound_score
    avg_sentiment = conversation_sentiment / len(conversation)
    return avg_sentiment, calculate_threshold(avg_sentiment)


def classify(sentence, analyzer):
    ss = analyzer.polarity_scores(sentence)
    # sentiment_analysis = [(k, ss[k]) for k in ss if k != 'compound']

    # label = max(sentiment_analysis, key=itemgetter(1))[0]
    label = calculate_threshold(ss['compound'])

    return ss['compound'], label
    # for k in ss:
    #     print('{0}: {1}, '.format(k, ss[k]))


def calculate_threshold(compound_value):
    # Set threshold for 'compound' score:
    # positive sentiment: compound score >= 0.05
    # neutral sentiment: (compound score > -0.05) and (compound score < 0.05)
    # negative sentiment: compound score <= -0.05
    label = ''
    if compound_value >= 0.03:
        label = "Positive"
    elif -0.03 < compound_value < 0.03:
        label = "Neutral"
    elif compound_value <= -0.03:
        label = "Negative"

    return label


def main():
    dir_path = r"./Spongebob NLP"
    conversations = read_files_to_dict(dir_path)

    sid = SentimentIntensityAnalyzer()

    character = 'Mrs. Puff'
    conv = get_all_character_conversations(character, conversations)

    character_all_conversations = conversations[character]
    avg_score, label = classify_conversation(character_all_conversations, sid)
    print('Sentiment of character {0} is: {1} with avg_score of {2}'.format(character, label, avg_score))

    for key, conversation in conv.iteritems():
        label = ''
        conversation_sentiments = 0.0
        # sentence = preprocess(sentence)
        avg_score, label = classify_conversation(conversation, sid)
        # sentence = sentence.encode('ascii', 'ignore').decode('ascii')
        # print('Sentiment of [{0}]: >>> {1}'.format(sentence, label))
        print('Sentiment of conversation {0} is: {1}'.format(key, label))


if __name__ == '__main__':
    main()
