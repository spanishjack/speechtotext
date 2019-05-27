#!/usr/bin/env python
# coding: utf-8

from __future__ import unicode_literals
import youtube_dl

video_url='https://www.youtube.com/watch?v=ouRbkBAOGEw'

ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
}
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download([video_url])

import os
import pydub
from pydub import AudioSegment

directory_path = '/Users/jhuck/Documents/text_to_speech'
input_file_name = 'JFK - We choose to go to the Moon, full length-ouRbkBAOGEw.mp3'
output_file_name = 'JFK - We choose to go to the Moon, full length-ouRbkBAOGEw.wav'

#initialize our audio segment and set channels to mono
audio = AudioSegment.from_file(os.path.join(directory_path, input_file_name), format = 'mp3')
audio = audio.set_channels(1)

#export audio to wav format
audio.export(os.path.join(directory_path, output_file_name), format = 'wav')

import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(directory_path, 'My First Project-02d9aeddc2f8.json')

from google.cloud import storage

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))

gc_storage_bucket_name = your_storage_bucket_name

upload_blob(gc_storage_bucket_name ,os.path.join(directory_path, output_file_name) ,output_file_name)

def transcribe_gcs(gcs_uri):
    """Asynchronously transcribes the audio file specified by the gcs_uri."""
    from google.cloud import speech
    from google.cloud.speech import enums
    from google.cloud.speech import types
    client = speech.SpeechClient()

    audio = types.RecognitionAudio(uri=gcs_uri)
    config = types.RecognitionConfig(
        encoding = enums.RecognitionConfig.AudioEncoding.LINEAR16,
        #sample_rate_hertz=16000,
        language_code='en-US',
        enable_automatic_punctuation=True)

    operation = client.long_running_recognize(config, audio)
    return_list = []
    print('Waiting for operation to complete...')
    response = operation.result(timeout=400)

    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    for result in response.results:
        #The first alternative is the most likely one for this portion.
        print(u'Transcript: {}'.format(result.alternatives[0].transcript))
        print('Confidence: {}'.format(result.alternatives[0].confidence))
        return_list.append(result.alternatives[0].transcript)
        
    return return_list

audio_file_uri = 'gs://' + gc_storage_bucket + '/' + output_file_name

output_text = transcribe_gcs(audio_file_uri)

text_file_name = 'jfk_moon.txt'
with open(os.path.join(directory_path, text_file_name), 'w') as f:
    for x in output_text:
        f.write("%s\n" % x)

import nltk
from nltk.tokenize import sent_tokenize

#break out text into sentances
sentence_list = []

for line in output_text:
    if '.' not in line:
        sentence_list.append(line.strip())
    else:
        sent = sent_tokenize(line)
        for x in sent:
            sentence_list.append(x.strip())

print(sentence_list[:10])

from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

word_list = []
word_exclude_list = []
lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))
word_types = ['NOUN']
words_to_keep = ['united-states']

record_count = 0

for line in sentence_list:
    tokenized_text = nltk.word_tokenize(line)
    
    #tokenized text is array with format [word, word type]
    tagged_and_tokenized_text = nltk.pos_tag(tokenized_text,tagset='universal') 

    for token in tagged_and_tokenized_text:
        
        #For my use case i remove all non-alpha characters and lowered the text for uniformity. Modify for your use case
        token = [re.sub('[^a-zA-Z]+', '', token[0]).lower(),token[1]]
        
        if (token[1] in word_types and token[0] not in stop_words and len(token[0]) > 1) or token[0] in words_to_keep:
            word_list.append([record_count,lemmatizer.lemmatize(token[0])])

        else:
            word_exclude_list.append([record_count,token])
        
    record_count += 1

print(word_list[:10])

import itertools as it
import numpy as np

node_edge_list = []
node_list = []

record_count = 0

for word in word_list:
    sentence = ([row for row in word_list if record_count == row[0]])

    temp_list = []
    
    for token in sentence:
        temp_list.append(token[1])

    temp_cartesian_list = list(it.combinations(temp_list, 2))

    for i in temp_cartesian_list: 
        node_edge_list.append(i)
        
    record_count += 1

#Sort the list to make sure column 1 is always < column 2 regarding sort.
#Example, we don't want to count [(1,2), (2,1)]. We want [(1,2), (1,2)]
#This will impact the step when we aggregate the pairs to calculate the edge weights.
node_edge_list_array = np.array(node_edge_list)
node_edge_list_sorted_array = np.sort(node_edge_list_array, axis=1)
node_edge_list = node_edge_list_sorted_array.tolist()

temp_node_list = []

for i in node_edge_list:
    temp_node_list.append(i[0])

node_list = list(set(temp_node_list))

print(node_edge_list[:10])

import pandas as pd

node_edge_dataframe = pd.DataFrame(node_edge_list,columns=['node', 'edge'])
node_edge_dataframe_count = node_edge_dataframe.groupby(['node','edge']).size().reset_index(name='counts')

output_edge_list_with_counts = node_edge_dataframe_count.values.tolist()

print(output_edge_list_with_counts[:10])

import networkx as nx

undirected_weighted_graph = nx.Graph()

for i in output_edge_list_with_counts:
    undirected_weighted_graph.add_edge(i[0], i[1], weight = i[2])
    
print(nx.info(undirected_weighted_graph))

import community

graph_partition = community.best_partition(undirected_weighted_graph)
nx.set_node_attributes(undirected_weighted_graph, graph_partition, 'best_community')

output_file_name = 'network_output.gexf' 
nx.write_gexf(undirected_weighted_graph, os.path.join(directory_path, output_file_name))