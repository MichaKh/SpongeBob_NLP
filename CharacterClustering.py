import os
import nltk
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, Normalizer
from nltk.stem.porter import PorterStemmer


def read_files_to_dict(dir_path):
    """
    Read file in specified path
    :param file_path: Path of file to read
    :return:
    """
    conversations = {}
    for filename in os.listdir(dir_path):
        if filename.endswith(".txt") and '-' not in filename:
            file_path = os.path.abspath(os.path.join(dir_path, filename))
            with open(file_path, 'r') as f:
                x = f.readlines()
                content_unicode = unicode(x[0], encoding='utf-8', errors='replace')
                if not filename in conversations:
                    conversations[filename.split('.txt')[0]] = content_unicode
    return conversations


def tokenize(text):
    stems = []
    sentences = text.split('@')
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        stems.append(stem_tokens(tokens, PorterStemmer()))
    flat_list = [item for sublist in stems for item in sublist]
    return flat_list


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def get_num_capitalized_words(data):
    """
    Calculate for each tweet the number of words containing at least one capital letter.
    :param data: Data to be processed
    :return: Numpy matrix containing count of capitalized words
    """
    capitalized = []
    for tweet in data:
        capitalized_words = [word for word in tweet.split(' ') if not word.islower() and not word.isupper()]
        capitalized.append(len(capitalized_words))

    return np.array(capitalized).reshape(-1, 1)


def get_num_exclamation_marks(data):
    """
    Calculate for each tweet the number of exclamation marks.
    :param data: Data to be processed
    :return: Numpy matrix containing count of exclamation marks
    """
    exclamation = []
    for tweet in data:
        excl_marks = [word for word in tweet.split(' ') if '!' in word]
        exclamation.append(len(excl_marks))

    return np.array(exclamation).reshape(-1, 1)


def get_num_hyphen_marks(data):
    """
    Calculate for each tweet the number of hyphen marks.
    :param data: Data to be processed
    :return: Numpy matrix containing count of hyphen marks
    """
    hyphen = []
    for tweet in data:
        hyph_marks = [word for word in tweet.split(' ') if '-' in word]
        hyphen.append(len(hyph_marks))

    return np.array(hyphen).reshape(-1, 1)


def get_num_POS_tags(data, pos_tag):
    """
    Calculate for each tweet the number of verb POS tags.
    :param data: Data to be processed
    :return: Numpy matrix containing count of verb POS tags.
    """
    pos_count = []
    for tweet in data:
        tokens = nltk.word_tokenize(tweet)
        tags = nltk.pos_tag(tokens)
        counts = Counter([j for i, j in tags])
        total = sum(counts.values())
        # normalized_counts = dict((word, float(count) / total) for word, count in counts.items())
        normalized_verb_count = sum(count for pos, count in counts.iteritems() if pos.startswith(pos_tag))
        # verb_counts = sum(1 for word, pos in normalized_counts if word.startswith('VB'))
        pos_count.append(normalized_verb_count / total)

    return np.array(pos_count).reshape(-1, 1)


def get_length(data):
    """
    Calculate for each tweet in the dataset its length.
    :param data: Data to be processed
    :return: Numpy matrix containing length for each tweet
    """
    return np.array([len(conv) for conv in data]).reshape(-1, 1)


def extract_features(data, max_df, max_features, min_df, stopword):
    vectorizer = TfidfVectorizer(max_df=max_df, max_features=max_features,
                                 min_df=min_df, stop_words=stopword)

    X = vectorizer.fit_transform(data)
    return X


def extract_features_pipeline():
    cl_pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text', Pipeline([
                ('vectorizer', CountVectorizer(input='content',
                                               analyzer='word',
                                               stop_words='english',
                                               lowercase=False,
                                               tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('svd', TruncatedSVD(3)),
                ('normalizer', Normalizer(copy=False))
            ])),
            ('length', Pipeline([
                ('count', FunctionTransformer(get_length, validate=False)),
            ])),
            ('capitalized', Pipeline([
                ('count', FunctionTransformer(get_num_capitalized_words, validate=False)),
            ])),
            ('exclamation', Pipeline([
                ('count', FunctionTransformer(get_num_exclamation_marks, validate=False)),
            ])),
            ('hyphen', Pipeline([
                ('count', FunctionTransformer(get_num_hyphen_marks, validate=False)),
            ])),
            ('verb_pos', Pipeline([
                ('count', FunctionTransformer(func=get_num_POS_tags, kw_args={'pos_tag': 'VB'}, validate=False)),
            ])),
        ],
            # weight components in FeatureUnion
            transformer_weights={
                'text': 0.8,
                'length': 0.5,
                'capitalized': 1.0,
                'exclamation': 0.6,
                'hyphen': 0.5,
                'verb_pos': 0.3,
            },
        ))])

    return cl_pipeline


def cluster(pipeline, data, tfidf_matrix, clusterer):
    pipeline.steps.append(['clusterer', clusterer])
    pipeline.fit(data.values())
    clusters = pipeline.steps[-1][1].labels_.tolist()
    str_clusters = print_clusters(data, clusters)

    # clusterer.fit(tfidf_matrix)
    # clusters = clusterer.labels_.tolist()
    # str_clusters = print_clusters(data, clusters)
    return clusters, str_clusters


def print_clusters(data, clusters):
    cluster_topic_pair = []
    str_clusters = ''
    # for i in range(n_cluster):
    for i in range(len(set(clusters))):
        for indx, topic in enumerate(clusters):
            if topic == i:
                cluster_topic_pair.append(['Cluster %d: ' % topic, '{0}'.format(data.keys()[indx])])

                str_clusters += cluster_topic_pair[-1][0]
                str_clusters += ''.join(cluster_topic_pair[-1][1])
                str_clusters += '\n'
    return str_clusters


def main():
    dir_path = r"./Spongebob NLP"
    data = read_files_to_dict(dir_path)
    tfidf_matrix = extract_features(data, max_df=100,
                                    min_df=1,
                                    max_features=10000,
                                    stopword='english')
    pipe = extract_features_pipeline()

    n_clusters = 4
    km = KMeans(n_clusters=n_clusters,
                init='k-means++',
                max_iter=100,
                n_init=1,
                verbose=0)

    eps = 0.6
    min_samples = 2
    dbs = DBSCAN(eps, min_samples, metric='euclidean')

    clusters, str_clusters = cluster(pipe, data=data, tfidf_matrix=tfidf_matrix, clusterer=km)

    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    print('Number of clusters is: %s' % n_clusters)

    print(print_clusters(data, clusters))


if __name__ == '__main__':
    main()
