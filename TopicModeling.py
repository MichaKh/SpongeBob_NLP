import os
import re
from collections import OrderedDict
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
import gensim
from gensim import corpora, similarities
import pyLDAvis.gensim
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


def read_files_to_dict(dir_path):
    """
    Read file in specified path
    :param file_path: Path of file to read
    :return:
    """
    conversations = {}
    for filename in os.listdir(dir_path):
        if filename.endswith(".txt"):
            file_path = os.path.abspath(os.path.join(dir_path, filename))
            with open(file_path, 'r') as f:
                x = f.readlines()
                content_unicode = unicode(x[0], encoding='utf-8', errors='replace')
                if not filename in conversations:
                    conversations[filename.split('.txt')[0]] = content_unicode
    return conversations


def split_conversation_to_sentences(conversation):
    sentences = []
    if conversation:
        sentences = conversation.split('@')
    return sentences


def preprocess(doc, stopwords_characters):
    stopwords_characters = [x.lower() for x in stopwords_characters]
    stop = set(stopwords.words('english')).union(stopwords_characters)
    exclude = set(string.punctuation)
    digits = '0123456789'
    # lemma = WordNetLemmatizer()
    # stemma = stemmer()
    # stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    # punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    # # digits_free = ''.join(d for d in punc_free if d not in digits)
    # normalized = ''.join(d for d in punc_free if d not in digits)
    # # stemmed = " ".join(stemma.stem(word) for word in digits_free.split())
    # # normalized = " ".join(lemma.lemmatize(word) for word in stemmed.split())
    #
    # return normalized

    tokenized_text = word_tokenize(doc.lower())
    cleaned_text = [t for t in tokenized_text if t not in stop and re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', t)]
    cleaned_text = ' '.join(w.strip('.\\?@\'~{}()') for w in cleaned_text)

    # Filter terms with the following POS tags
    tokens = nltk.word_tokenize(cleaned_text)
    tags = nltk.pos_tag(tokens)

    dt_tags = [t for t in tags if t[1] not in ["DT", "MD", "TO"
                                               "CD", "PDT", "WDT",
                                               "EX", "CC", "RP", "IN",
                                               "RB", "RBR", "RBS",
                                               "VBZ", "VB", "VBD", "VBG", "VBN", "VBP"]]
    filtered_text = [w[0] for w in dt_tags]
    return ' '.join(filtered_text)


def get_doc_term_matrix(doc_clean):
    # Creating the term dictionary of our corpus, where every unique term is assigned an index.
    dictionary = corpora.Dictionary(doc_clean)

    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    return dictionary, doc_term_matrix


def get_lda_topics(num_topics, passes, dictionary, doc_term_matrix, n_top_terms):
    # Creating the object for LDA model using gensim library
    Lda = gensim.models.ldamodel.LdaModel

    # Running and Trainign LDA model on the document term matrix.
    ldamodel = Lda(doc_term_matrix, num_topics=num_topics, id2word=dictionary, passes=passes)

    topic_words = []

    for i in range(num_topics):
        tt = ldamodel.get_topic_terms(i, n_top_terms)
        topic_words.append([dictionary[pair[0]] for pair in tt])

    return ldamodel, topic_words


def get_most_similar_doc(text_to_compare, lda_model, corpus, dictionary):
    lda_index = gensim.similarities.MatrixSimilarity(lda_model[corpus])

    bow = dictionary.doc2bow(preprocess(text_to_compare, ['']).split())
    # Let's perform some queries
    similarities = lda_index[lda_model[bow]]
    # Sort the similarities
    similarities = sorted(enumerate(similarities), key=lambda item: -item[1])

    # Top most similar documents:
    # print(similarities[:10])

    # Let's see what's the most similar document
    document_id, similarity = similarities[1]

    return document_id, similarity


def main():
    dir_path = r"./Spongebob NLP"
    conversations = read_files_to_dict(dir_path)

    normalized_conversations = OrderedDict()

    for doc_name, doc in conversations.iteritems():
        normalized = preprocess(doc=doc, stopwords_characters=conversations.keys())
        normalized_conversations[doc_name] = normalized.split()

    dictionary, doc_term_matrix = get_doc_term_matrix(doc_clean=normalized_conversations.values())

    ldamodel, topics = get_lda_topics(num_topics=5,
                                      passes=50,
                                      dictionary=dictionary,
                                      doc_term_matrix=doc_term_matrix,
                                      n_top_terms=20)

    print(ldamodel.print_topics(num_topics=5, num_words=3))

    # pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(ldamodel, doc_term_matrix, dictionary, mds='tsne')
    # pyLDAvis.display(vis)
    # pyLDAvis.show(vis)

    character_to_compare = 'Squidward'
    doc_id, similarity = get_most_similar_doc(conversations[character_to_compare], ldamodel, doc_term_matrix, dictionary)
    print('The most similar document to {0} is: {1}; \n{2}'.format(character_to_compare,
                                                            normalized_conversations.items()[doc_id][0],
                                                            " ".join(normalized_conversations.items()[doc_id][1][:1000])))

    #################################

    character = 'Mr. Krabs-Plankton'

    characters_names = conversations
    normalized = preprocess(doc=conversations[character], stopwords_characters=characters_names)

    dictionary, doc_term_matrix = get_doc_term_matrix(doc_clean=[normalized.split()])

    ldamodel, topics = get_lda_topics(num_topics=3,
                                      passes=50,
                                      dictionary=dictionary,
                                      doc_term_matrix=doc_term_matrix,
                                      n_top_terms=20)

    print(ldamodel.print_topics(num_topics=3, num_words=3))

    # pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(ldamodel, doc_term_matrix, dictionary)
    # pyLDAvis.display(vis)
    pyLDAvis.show(vis)


if __name__ == '__main__':
    main()
