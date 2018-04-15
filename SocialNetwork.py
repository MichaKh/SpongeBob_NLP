import os
import string
import nltk
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
import SentimentAnalysis


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


def count_ne_normalized(tagging):
    """
    Count the number of occurrences of each named entity in a given tagging.
    :param tagging: Given tagging of sentence.
    :return: Dictionary structure of counts.
    """
    length = len(tagging)
    if tagging:
        ne_counts = nltk.Counter(tagging)
        normalized = {}
        for key in ne_counts:
            normalized[key] = float(ne_counts[key]) / length
        return normalized
    else:
        return None


def count_ne(tagging):
    """
        Count the number of occurrences of each named entity in a given tagging.
        :param tagging: Given tagging of sentence.
        :return: Dictionary structure of counts.
        """
    if tagging:
        ne_counts = nltk.Counter(tagging)
        counts = {}
        for key in ne_counts:
            counts[key] = ne_counts[key]
        return counts
    else:
        return None


def pos_tag(s):
    """
    Tag the given text to its corresponding Part Of Speech (POS) tags.
    Use the NLTK package.
    :param s: String sentence to tag.
    :return: List of POS tags.
    """
    tokenized = nltk.word_tokenize(s)
    tagged = nltk.pos_tag(tokenized)
    return tagged


def recognize_ne(s):
    """
    Recognize named entities in given sentence.
    Use the NLTK package.
    :param s: String sentence to tag.
    :return: Tree structure of NE recognition.
    """
    s_t = pos_tag(s)
    ne_tree = nltk.ne_chunk(s_t, binary=False)
    # iob_tags = nltk.tree2conlltags(ne_tree)
    entities = []
    entities.extend(extract_entity_names(ne_tree))
    return entities


def extract_entity_names(ne_tree):
    """
    Traverse the NE tree structure to extract the named entities.
    :param ne_tree: Tree structure of NE's.
    :return: List of the NEs and their tag.
    """
    ne_in_sent = []
    for subtree in ne_tree:
        if type(subtree) == nltk.Tree:  # If subtree is a noun chunk, i.e. NE != "O"
            ne_label = subtree.label()
            ne_string = " ".join([token for token, pos in subtree.leaves()])
            ne_in_sent.append((ne_string, ne_label))
    return ne_in_sent


def get_num_of_PERSON_NE(ne_tagging, character):
    # length = len(ne_tagging)
    count = 0
    # observed_tags = [t[1] for t in ne_tagging]
    all_PERSON_ne = [t[0] for t in ne_tagging if t[1] == 'PERSON']
    ne_counts = count_ne(all_PERSON_ne)
    if character in ne_counts:
        count = ne_counts[character]
    return count


def build_sentiment_graph(nodes, edges):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    # for node in nodes:
    #     G.add_node(node, name=node)
    for edge in edges:
        u = edges[edge]['from']
        v = edges[edge]['to']
        weight = edges[edge]['weight']
        color = edges[edge]['color']
        width = edges[edge]['conv_length']
        G.add_edge(u, v, weight=weight, color=color, edge_width=width)
    return G


def create_nodes(characters_dict):
    nodes = []
    for name, size in characters_dict.iteritems():
        nodes.append((name, dict(size=size, name=name)))
    return nodes


def create_edges(conversations, characters):
    edges = {}
    sid = SentimentIntensityAnalyzer()
    for character in characters:
        char_conversations = SentimentAnalysis.get_all_character_conversations(character=character, conversations=conversations)
        for name, conversation in char_conversations.iteritems():
            splitted_conversation = conversation.split('@')
            avg_score, label = SentimentAnalysis.classify_conversation(splitted_conversation, analyzer=sid)
            if '-' in name:
                first_character, second_character = name.split('-')
                if first_character in characters and second_character in characters:
                    edges[name] = {}
                    edges[name]['from'] = first_character
                    edges[name]['to'] = second_character
                    edges[name]['weight'] = format(avg_score, '.2f')
                    edges[name]['conv_length'] = len(splitted_conversation)
                    if label == 'Positive':
                        edges[name]['color'] = 'g'
                    elif label == 'Negative':
                        edges[name]['color'] = 'r'
                    else:
                        edges[name]['color'] = 'b'
    return edges


def draw_graph(G):
    plt.figure(figsize=(10, 8))

    # node_color = [node['color'] for node in G.nodes()]
    color = nx.get_edge_attributes(G, 'color')
    edge_width = nx.get_edge_attributes(G, 'edge_width')
    names = nx.get_node_attributes(G, 'name')
    size = nx.get_node_attributes(G, 'size')

    edge_colors = [color[edge] for edge in G.edges()]
    edge_width = [float(edge_width[edge]) for edge in G.edges()]
    normalized_edge_width = [width / max(edge_width) * 10 for width in edge_width]

    node_size = [size[node] * 15000 for node in G.nodes()]
    edge_labels = dict([((u, v,), d['weight'])
                        for u, v, d in G.edges(data=True)])

    pos = nx.circular_layout(G, scale=3)
    pos['Spongebob'] = np.array([0, 0])  # position Spongebob as the middle vertex

    # pos = nx.spring_layout(G)
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, font_size=8)
    nx.draw(G, pos=pos,
            node_color=range(len(G)),
            edge_color=edge_colors,
            node_size=node_size,
            width=normalized_edge_width,
            with_labels=True)
    # pylab.show()
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    dir_path = r"./Spongebob NLP"
    conv = read_files_to_dict(dir_path)

    num_of_conv_sentences_count = {}
    num_of_char_sentences_count = {}

    person_count_dict = {}

    for character, conversation in conv.iteritems():
        if '-' in character:
            num_of_conv_sentences_count[character] = len(conversation.split('@'))

    for character, conversation in conv.iteritems():
        if '-' not in character:
            num_of_char_sentences_count[character] = len(conversation.split('@'))

    total_num_of_sentences = sum(num_of_conv_sentences_count.values())
    normalized_sentences_count = {k: v / float(total_num_of_sentences)*100 for k, v in num_of_conv_sentences_count.iteritems()}
    sorted_normalized_sentence_count = sorted(normalized_sentences_count.items(), key=lambda x: x[1], reverse=True)
    for character, count in sorted_normalized_sentence_count:
            print('Proportion of sentences in conversation by {0} is {1}%'.format(character, count))

    total_num_of_sentences = sum(num_of_char_sentences_count.values())
    normalized_sentences_count = {k: v / float(total_num_of_sentences)*100 for k, v in num_of_char_sentences_count.iteritems()}
    sorted_normalized_sentence_count = sorted(normalized_sentences_count.items(), key=lambda x: x[1], reverse=True)
    for character, count in sorted_normalized_sentence_count:
        print('Proportion of sentences spoken by {0} is {1}%'.format(character, count))

    for key, conversation in conv.iteritems():
        if '-' not in key:
            conversation = conversation.replace('@', '')
            printable = set(string.printable)
            conversation = filter(lambda x: x in printable, conversation)
            ne_tagging = recognize_ne(conversation)

            for character in conv.keys():
                character_count = get_num_of_PERSON_NE(ne_tagging, character)
                if character in person_count_dict:
                    person_count_dict[character] += character_count
                else:
                    person_count_dict[character] = character_count

    # Normalize count of PERSON NEs
    total = sum(person_count_dict.itervalues(), 0.0)
    normalized_person_count_dict = {k: v / total for k, v in person_count_dict.iteritems()}

    filtered_person_count_dict = {k: v for k, v in normalized_person_count_dict.iteritems() if '-' not in k
                                  and v >= 0 / float(total)}

    nodes = create_nodes(filtered_person_count_dict)
    character_nodes = [t[0] for t in nodes]
    edges = create_edges(conv, character_nodes)
    G = build_sentiment_graph(nodes, edges)
    draw_graph(G)


if __name__ == '__main__':
    main()
