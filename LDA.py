#!/usr/bin/env python
# coding: utf-8

# In[40]:


import dimcli
import pandas as pd
import requests
import json
import csv
import numpy as np
import re
import nltk
import gensim
import spacy
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
import os
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
np.random.seed(123)
import pickle
nltk.download('wordnet')
#get_ipython().run_line_magic('matplotlib', 'inline')
import dash
#import dash_core_components as dcc
from dash import dcc
#import dash_html_components as html
from dash import html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State




# In[41]:
def lemmatize_stemming(text):
    return WordNetLemmatizer().lemmatize(text, pos='v')
def preprocess_abstract(text):
    redundant = ['abstract', 'purpose', 'paper', 'goal', 'usepackage', 'cod']
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in redundant:
            result.append(lemmatize_stemming(token))
    return " ".join(result)
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
def split_into_ranks(array):
    ranks = []
    for value in array:
        for i, percentage in enumerate(np.arange(.1, 1.1, .1)):
            if value <= np.quantile(array, percentage):
                ranks.append(i + 1)
                break
    return ranks
def display_topics_list(model, feature_names, no_top_words):
    topic_list = []
    for topic_idx, topic in enumerate(model.components_):
        topic_list.append(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
    return topic_list
def create_top_list(data_frame, num_topics, threshold, filtered):
    top_5s = []
    the_filter = filtered[threshold][num_topics]
    for topic in range(num_topics):
        relevant = the_filter[the_filter[f'Topic{topic}'] != 0].index.to_list()
        to_append = data_frame[data_frame[f'{topic}_relevance'] > 0].reset_index()
        to_append = to_append[to_append['author'].isin(relevant)].reset_index()
        top_5s.append(to_append)
    return top_5s

def LDA_model(datafile, hdsi):
    df = pd.read_csv(datafile)
    print('hi')
    #df = pd.DataFrame.from_dict(data, orient="index").transpose()
    authors = df[['authors']]


    # In[43]:

    test = str(authors.loc[0][0])
    fg = list(eval(test))#[0]['first_name']
    lis = []
    lis2 = []
    for i in fg:
        if 'first_name' in i:
            first = i['first_name']
            last = i['last_name']
            full = first + " " + last
            #print(full)
            lis.append(full)
            ids = i['researcher_id']
            #print(ids)
            lis2.append(ids)


    # In[44]:


    #new list to collect names
    new = []
    #new list to collect corresponding ids
    new2 = []
    #looping through length of author column
    for i in range(len(authors)):
        #turning string of list of dictionaries into list of dictionaries
        temp = list(eval(authors.loc[i][0]))
        #names
        lis = []
        #ids
        lis2 = []
        #looping through the list of dictionaries
        for i in temp:
            if 'first_name' in i:
                first = i['first_name']
                last = i['last_name']
                #concatenating first and last name
                full = first + " " + last
                lis.append(full)
                #print(lis)
                ids = i['researcher_id']
                lis2.append(ids)
            else:
                lis.append(i)
                lis2.append(i)
        new.append(lis)
        new2.append(lis2)


    # In[45]:


    #adding new column, "names," to the original dataframe
    names = pd.Series(new)
    df['names'] = names.values


    # In[46]:


    #adding new column, "ids," to the original dataframe
    ids = pd.Series(new2)
    df['ids'] = ids.values


    # # Aggregate data by researcher-year

    # In[47]:


    #df2 = df.explode(['names', 'ids']).reset_index(drop=True)
    df2 = df.apply(pd.Series.explode).reset_index(drop=True)

    testing = df2['ids'].value_counts()
    #print(testing.to_string())


    # In[48]:


    hdsi = pd.read_csv(hdsi)
    #hdsi = pd.DataFrame.from_dict(hdsi, orient="index").transpose()
    faculty = hdsi[hdsi['Dimensions ID'] != 'no ID']['Dimensions ID']
    #manually adding professors since they do not have dimensions ids
    add = pd.Series(['Aaron McMillan Fraenkel', 'Justin Eldridge'])
    faculty = list(faculty.append(add))


    # In[49]:


    #cleaned out all names & ids that do not match our hdsi faculty list
    df3 = df2[df2.ids.isin(faculty)].reset_index()


    # In[51]:


    #df3['ids'].unique()


    # In[52]:


    df3['abstract'] = df3['abstract'].fillna('')


    # In[53]:
    stemmer = PorterStemmer()

    df3['abstract_processed'] = df3['abstract'].apply(preprocess_abstract)


    # In[54]:


    df3 = df3[df3['year'] >= 2015]
    counts = CountVectorizer().fit_transform(df3['abstract_processed'])
    authors = {}
    for author in df3.names.unique():
        authors[author] = {
            2015 : list(),
            2016 : list(),
            2017 : list(),
            2018 : list(),
            2019 : list(),
            2020 : list(),
            2021 : list()
        }
    for i, row in df3.iterrows():
        authors[row['names']][row['year']].append(row['abstract_processed'])


    # In[55]:


    all_docs = []
    missing_author_years = {author : list() for author in df3.names.unique()}
    for author, author_dict in authors.items():
        for year, documents in author_dict.items():
            if len(documents) == 0:
                missing_author_years[author].append(year)
                continue
            all_docs.append(" ".join(documents))
    len(all_docs)


    # In[57]:


    # initate LDA model
    countVec = CountVectorizer()
    counts = countVec.fit_transform(all_docs)
    names = countVec.get_feature_names()


    # In[58]:


    modeller = LatentDirichletAllocation(n_components=10, n_jobs=-1, random_state=123)
    result = modeller.fit_transform(counts)
    modeller2 = LatentDirichletAllocation(n_components=20, n_jobs=-1, random_state=123)
    result2 = modeller2.fit_transform(counts)
    modeller3 = LatentDirichletAllocation(n_components=30, n_jobs=-1, random_state=123)
    result3 = modeller3.fit_transform(counts)
    modeller4 = LatentDirichletAllocation(n_components=40, n_jobs=-1, random_state=123)
    result4 = modeller4.fit_transform(counts)
    modeller5 = LatentDirichletAllocation(n_components=50, n_jobs=-1, random_state=123)
    result5 = modeller5.fit_transform(counts)

    models = {'10':modeller,'20':modeller2,'30':modeller3,'40':modeller4,'50':modeller5}
    results = {'10':result,'20':result2,'30':result3,'40':result4,'50':result5}


    # In[59]:


    topicnames = {
        num_topics : ["Topic" + str(i) for i in range(num_topics)] for num_topics in range(10, 60, 10)
    }

    # index names
    docnames = ["Doc" + str(i) for i in range(len(all_docs))]

    # Make the pandas dataframe
    df_document_topic = {
        num_topics : pd.DataFrame(results[f'{num_topics}'], columns=topicnames[num_topics], index=docnames) for num_topics in range(10, 60, 10)
    }

    # Get dominant topic for each document
    dominant_topic = {
        num_topics : np.argmax(df_document_topic[num_topics].values, axis=1) for num_topics in range(10, 60, 10)
    }

    for num_topics, df in df_document_topic.items():
        df['dominant_topic'] = dominant_topic[num_topics]


    # In[60]:


    author_list = []
    year_list = []
    for author in authors.keys():
        for i in range(7):
            if (2015 + i) not in missing_author_years[author]:
                author_list.append(author)
                year_list.append(2015 + i)

    for df in df_document_topic.values():
        df['author'] = author_list
        df['year'] = year_list


    # In[61]:


    averaged = {
        num_topics : df_document_topic[num_topics].groupby('author').mean().drop(['dominant_topic', 'year'], axis=1) for num_topics in df_document_topic.keys()
    }

    filtered = {
        threshold : {num_topics : averaged[num_topics].mask(averaged[num_topics] < threshold, other=0) for num_topics in averaged.keys()} for threshold in [.1]
    }


    # In[62]:


    labels = {}
    for num_topics in range(10, 60, 10):
        labels[num_topics] = filtered[.1][num_topics].index.to_list()
        labels[num_topics].extend(filtered[.1][num_topics].columns.to_list())


    sources = {threshold : {} for threshold in [.1]}
    targets = {threshold : {} for threshold in [.1]}
    values = {threshold : {} for threshold in [.1]}

    for threshold in [.1]:
        for num_topics in range(10, 60, 10):
            curr_sources = []
            curr_targets = []
            curr_values = []
            index_counter = 0
            for index, row in filtered[threshold][num_topics].iterrows():
                for i, value in enumerate(row):
                    if value != 0:
                        curr_sources.append(index_counter)
                        curr_targets.append(108 + i)
                        curr_values.append(value)
                index_counter += 1
            sources[threshold][num_topics] = curr_sources
            targets[threshold][num_topics] = curr_targets
            values[threshold][num_topics] = curr_values

    positions = {
        num_topics : {label : i for i, label in enumerate(labels[num_topics])} for num_topics in averaged.keys()
    }


    # In[63]:

    final_values = {threshold : {} for threshold in [.1]}

    for threshold in [.1]:
        for num_topics in range(10, 60, 10):
            curr_values_array = np.array(values[threshold][num_topics])
            final_values[threshold][num_topics] = split_into_ranks(curr_values_array)


    # In[64]:


    # In[65]:


    link_labels = {}
    for num_topics in range(10, 60, 10):
        link_labels[num_topics] = labels[num_topics].copy()
        link_labels[num_topics][50:] = display_topics_list(models[f'{num_topics}'], names, 10)


    # In[66]:


    counts = CountVectorizer().fit_transform(df3['abstract_processed'])
    transformed_list = []
    for model in models.values():
        transformed_list.append(model.transform(counts))


    # In[67]:


    dataframes = {threshold : {} for threshold in [.1]}
    for i, matrix in enumerate(transformed_list):
        for threshold in [.1]:
            df = pd.DataFrame(matrix)
            df.mask(df < threshold, other=0, inplace=True)
            df['author'] = df3['names']
            df['year'] = df3['year']
            df['citations'] = df3['times_cited'] + 1

            # noralization of citations: Scaling to a range [0, 1]
            df['citations_norm'] = df.groupby(by=['author', 'year'])['citations'].apply(lambda x: (x-x.min())/(x.max()-x.min()))#normalize_by_group(df=df, by=['author', 'year'])['citations']
            df['abstract'] = df3['abstract']
            df['title'] = df3['title']
            df.fillna(1, inplace=True)

            #alpha weight parameter for weighting importance of citations vs topic relation
            alpha = .75
            for topic_num in range((i+1) * 10):
                df[f'{topic_num}_relevance'] = alpha * df[topic_num] + (1-alpha) * df['citations_norm']
            dataframes[threshold][(i+1) * 10] = df


    # In[68]:
    tops = {
        threshold : {num_topics : create_top_list(dataframes[threshold][num_topics], num_topics, threshold, filtered) for num_topics in range(10, 60, 10)} for threshold in [.1]
    }


    # In[69]:
    # sankey diagrams for diff numbers of topics

    heights = {
      10 : 1000,
      20 : 1500,
      30 : 2000,
      40 : 2500,
      50 : 3000
    }

    figs = {threshold : {} for threshold in [.1]}
    for threshold in [.1]:
        for num_topics in range(10, 60, 10):
            fig = go.Figure(data=[go.Sankey(
                node = dict(
                    pad = 15,
                    thickness = 20,
                    line = dict(color = 'black', width = 0.5),
                    label = labels[num_topics],
                    color = ['#666699' for i in range(len(labels[num_topics]))],
                    customdata = link_labels[num_topics],
                    hovertemplate='%{customdata} Total Flow: %{value}<extra></extra>'
                ),
                link = dict(
                    color = ['rgba(204, 204, 204, .5)' for i in range(len(sources[threshold][num_topics]))],
                    source = sources[threshold][num_topics],
                    target = targets[threshold][num_topics],
                    value = final_values[threshold][num_topics]
                )
            )])
            fig.update_layout(title_text="Author Topic Connections", font=dict(size = 10, color = 'white'), height=heights[num_topics], paper_bgcolor="black", plot_bgcolor='black')
            figs[threshold][num_topics] = fig


    # In[71]:


    top_words = {
        10 : display_topics_list(models['10'], names, 10),
        20 : display_topics_list(models['20'], names, 10),
        30 : display_topics_list(models['30'], names, 10),
        40 : display_topics_list(models['40'], names, 10),
        50 : display_topics_list(models['50'], names, 10)
    }

    combined = pd.read_csv(datafile)
    #combined[combined.title == 'Unperturbed: spectral analysis beyond Davis-Kahan'].abstract


    # In[72]:


    locations = {}
    for i, word in enumerate(names):
        locations[word] = i


    # In[ ]:


    from itertools import chain
    threshold = .1
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

    app.layout = html.Div([
      dbc.Row([
          dcc.Dropdown(
            id='graph-dropdown',
            placeholder='select number of LDA topics',
            options=[{'label' : f'{i} Topic Model', 'value' : i} for i in range(10, 60, 10)],
            style={
              'color' : 'black',
              'background-color' : '#666699',
              'width' : '200%',
              'align-items' : 'left',
              'justify-content' : 'left',
              'padding-left' : '15px'
            },
            value=10
          )
      ]),
      dbc.Row([
        dbc.Col(html.Div([
          dcc.Graph(
            id = 'graph',
            figure = figs[.1][10]
          )
          ],
          style={
            'height' : '100vh',
            'overflow-y' : 'scroll'
          }
        )
        ),
          dbc.Col(html.Div([dbc.Col([
            dcc.Dropdown(
              id='dropdown_menu',
              placeholder='Select a topic',
              options=[{'label' : f'Topic {topic}: {top_words[10][topic]}', 'value' : topic} for topic in range(10)],
              style={
                'color' : 'black',
                'background-color' : 'white'
              }
            ),
            dcc.Dropdown(
              id='researcher-dropdown',
              placeholder='Select Researchers',
              options=[{'label' : f'{researcher}', 'value' : f'{researcher}'} for researcher in set(author_list)],
              style={
                'color' : 'black',
                'background-color' : 'white'
              }
            )]),
            dbc.Col(
              dcc.Dropdown(
                id='word-search',
                placeholder='Search by word',
                options=[{'label' : word, 'value' : word} for word in names],
                style={
                  'color' : 'black',
                  'background-color' : 'white'
                },
                value=[],
                multi=True
              )
            ),
            html.Div(
              id='paper_container',
              children=[
                html.P(
                  children=['Top 5 Papers'],
                  id='titles_and_authors',
                  draggable=False,
                  style={
                    'font-size' :'150%',
                    'font-family' : 'Verdana'
                  }
                ),
              ],
            ),
          ],
            style={
              'height' : '100vh',
              'overflow-y' : 'scroll'
            }
          )
          )
        ]
      )]
    )

    @app.callback(
      Output('titles_and_authors', 'children'),
      Output('researcher-dropdown', 'options'),
      Input('dropdown_menu', 'value'),
      Input('graph-dropdown', 'value'),
      Input('researcher-dropdown', 'value'),
      Input('word-search', 'value')
    )
    def update_p(topic, num_topics, author, words):
      if len(words) != 0:
        doc_vec = np.zeros((1, len(names)))
        for word in words:
          doc_vec[0][locations[word]] += 1
        relations = np.round(models[f'{num_topics}'].transform(doc_vec), 5).tolist()[0]
        pairs = [(i, relation) for i, relation in enumerate(relations)]
        pairs.sort(reverse=True, key=lambda x: x[1])
        to_return = [[html.Br(), f'Topic{pair[0]}: {pair[1]}', html.Br()] for pair in pairs]
        return list(chain(*to_return)), [{'label' : f'{researcher}', 'value' : f'{researcher}'} for researcher in set(author_list)]

      if topic == None and author == None:
        return ['Make a selection'], [{'label' : f'{researcher}', 'value' : f'{researcher}'} for researcher in set(author_list)]

      if topic != None and author == None:
        df = tops[threshold][num_topics][topic]
        df_authors = df.author.unique()
        max_vals = df.groupby('author').max()[f'{topic}_relevance']

        to_return = [[f'{name}:', html.Br(),
          f'{df[df[f"{topic}_relevance"] == max_vals.loc[name]]["title"].to_list()[0]}',
          html.Details([html.Summary('Abstract'),
                        html.Div(combined[combined.title == f'{df[df[f"{topic}_relevance"] == max_vals.loc[name]]["title"].to_list()[0]}'].abstract)],
                        style={
                          'font-size' :'80%',
                          'font-family' : 'Verdana'}),
          html.Br()] for i, name in enumerate(max_vals.index)]
        return list(chain(*to_return)), [{'label' : f'{researcher}', 'value' : f'{researcher}'} for researcher in tops[threshold][num_topics][topic].author.unique()]

      if topic == None and author != None:
        to_return = []
        for topic_num in range(num_topics):
          df = tops[threshold][num_topics][topic_num]
          if author in df.author.unique():
            max_vals = df.groupby('author').max()[f'{topic_num}_relevance']

            to_return.append([f'Topic {topic_num}:', html.Br(),
              f'{df[df[f"{topic_num}_relevance"] == max_vals.loc[author]]["title"].to_list()[0]}',
              html.Details([html.Summary('Abstract'),
                            html.Div(combined[combined.title == f'{df[df[f"{topic_num}_relevance"] == max_vals.loc[author]]["title"].to_list()[0]}'].abstract)],
                            style={
                              'font-size' :'80%',
                              'font-family' : 'Verdana'},
                            ),
              html.Br()])
        return list(chain(*to_return)), [{'label' : f'{researcher}', 'value' : f'{researcher}'} for researcher in set(author_list)]

      if topic != None and author != None:
        df = tops[threshold][num_topics][topic]
        df = df[df['author'] == author]
        df.sort_values(by=f'{topic}_relevance', ascending=False, inplace=True)
        titles = df.head(10)['title'].to_list()

        to_return = [
          [f'{i} : {title}',
          html.Details([html.Summary('Abstract'),
                        html.Div(combined[combined.title == title].abstract)],
                        style={
                          'font-size' :'80%',
                          'font-family' : 'Verdana'}),
          html.Br()] for i, title in enumerate(titles)]
        return list(chain(*to_return)), [{'label' : f'{researcher}', 'value' : f'{researcher}'} for researcher in tops[threshold][num_topics][topic].author.unique()]



    @app.callback(
      [Output('graph', 'figure'), Output('dropdown_menu', 'options')],
      [Input('graph-dropdown', 'value'), Input('dropdown_menu', 'value'), Input('researcher-dropdown', 'value'), Input('word-search', 'value')],
      State('graph', 'figure')
    )
    def update_graph(value, topic, author, words, previous_fig):
      if len(previous_fig['data'][0]['node']['color']) != value + 108:
        figs[threshold][value].update_traces(node = dict(color = ['#666699' for i in range(len(labels[value]))]), link = dict(color = ['rgba(204, 204, 204, .5)' for i in range(len(sources[threshold][value]))]))
        return figs[threshold][value], [{'label' : f'Topic {topic}: {top_words[value][topic]}', 'value' : topic} for topic in range(value)]

      if len(words) != 0:
        doc_vec = np.zeros((1, len(names)))
        for word in words:
          doc_vec[0][locations[word]] += 1
        relations = np.round(models[f'{value}'].transform(doc_vec), 3).tolist()[0]
        opacity = {(i+108) : relation for i, relation in enumerate(relations) if relation > .1}
        node_colors = ['#666699' if (i not in opacity.keys()) else f'rgba(255, 255, 0, {opacity[i]})' for i in range(len(labels[value]))]
        valid_targets = [positions[value][f'Topic{i-108}'] for i in opacity.keys()]
        link_colors = ['rgba(204, 204, 204, .5)' if target not in valid_targets else f'rgba(255, 255, 0, .5)' for target in targets[threshold][value]]
        figs[threshold][value].update_traces(node = dict(color = node_colors), link = dict(color = link_colors)),
        return figs[threshold][value], [{'label' : f'Topic {topic}: {top_words[value][topic]}', 'value' : topic} for topic in range(value)]


      if topic == None and author == None:
        figs[threshold][value].update_traces(node = dict(color = ['#666699' for i in range(len(labels[value]))]), link = dict(color = ['rgba(204, 204, 204, .5)' for i in range(len(sources[threshold][value]))]))
        return figs[threshold][value], [{'label' : f'Topic {topic}: {top_words[value][topic]}', 'value' : topic} for topic in range(value)]

      if topic != None and author == None:
        node_colors = ['#666699' if (i != positions[value][f'Topic{topic}']) else '#ffff00' for i in range(len(labels[value]))]
        link_colors = ['rgba(204, 204, 204, .5)' if target != positions[value][f'Topic{topic}'] else 'rgba(255, 255, 0, .5)' for target in targets[threshold][value]]
        figs[threshold][value].update_traces(node = dict(color = node_colors), link = dict(color = link_colors))
        return figs[threshold][value], [{'label' : f'Topic {topic}: {top_words[value][topic]}', 'value' : topic} for topic in range(value)]

      if topic == None and author != None:
        node_colors = ['#666699' if (i != positions[value][author]) else '#ffff00' for i in range(len(labels[value]))]
        link_colors = ['rgba(204, 204, 204, .5)' if source != positions[value][author] else 'rgba(255, 255, 0, .5)' for source in sources[threshold][value]]
        figs[threshold][value].update_traces(node = dict(color = node_colors), link = dict(color = link_colors))
        return figs[threshold][value], [{'label' : f'Topic {topic}: {top_words[value][topic]}', 'value' : topic} for topic in range(value)]

      if topic != None and author != None:
        node_colors = ['#666699' if (i != positions[value][author] and i != positions[value][f'Topic{topic}']) else '#ffff00' for i in range(len(labels[value]))]
        link_colors = ['rgba(204, 204, 204, .5)' if (source != positions[value][author] or target != positions[value][f'Topic{topic}']) else 'rgba(255, 255, 0, .5)' for source, target in zip(sources[threshold][value], targets[threshold][value])]
        figs[threshold][value].update_traces(node = dict(color = node_colors), link = dict(color = link_colors))
        return figs[threshold][value], [{'label' : f'Topic {topic}: {top_words[value][topic]}', 'value' : topic} for topic in range(value)]

    @app.callback(
      Output('researcher-dropdown', 'value'),
      Input('dropdown_menu', 'value'),
      State('dropdown_menu', 'value')
    )
    def reset_author(topic, previous):
      if topic != previous:
        return None



    app.run_server()


# In[ ]:
