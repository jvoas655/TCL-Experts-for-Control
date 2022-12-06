'''The following sentences were generated using OpenAI Chatbot with the following prompt: 
"Generate a set of 50 unique diverse sentences, and label the topic of each sentence."
We run the prompt several times until we have over 100 unique sentences, topic pairs
'''

from collections import defaultdict
import random

def create_text_topic_pairs(file_path):
    file = open(file_path, 'r')
    lines = file.readlines()
    
    topic_to_sentence = defaultdict(set)
    for line in lines:
        sentence, topic = line.split(" (")
        topic = topic.rstrip(")\n")
        topic_to_sentence[topic].add(sentence)
    
    topic_sentence_pairs = []
    for topic, sentences in topic_to_sentence.items():
        for sentence in sentences:
            topic_sentence_pairs.append((topic, sentence))
    random.shuffle(topic_sentence_pairs)
    return topic_sentence_pairs   

topic_sentence_pairs = create_text_topic_pairs("src/eval/sentences_with_labels.txt")
print(f"Number of unique sentence, topic pairs: {len(topic_sentence_pairs)}")
