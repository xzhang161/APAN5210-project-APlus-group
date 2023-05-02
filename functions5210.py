#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import itertools
import pandas as pd
import jellyfish  
import nltk  
import string
import re
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import matplotlib.pyplot as plt


# ### 1, Load datasets

# In[2]:


def load_data(left, right):
    df_left = pd.read_csv(left)
    df_right = pd.read_csv(right)
    return df_left, df_right


# ### 2, Data analysis and visualizations

# In[3]:


def analysis(df):
    print("Shape:", df.shape)
    
    print("\nColumn names:")
    for column in df.columns:
        print(column)
        
    print("\nData types:")
    print(df.dtypes)

    print("\nNull values:")
    print(df.isnull().sum())


# In[4]:


def visualization(df):
    # city count
    city_count = df['city'].value_counts().head(10)
    city_count.plot.bar()
    plt.xlabel('cities')
    plt.ylabel('Count')
    plt.title('Number of Top 10 cities in the dataset')
    plt.show()
    
    # state count
    state_count = df['state'].value_counts().head(5)
    state_count.plot.bar()
    plt.xlabel('states')
    plt.ylabel('Count')
    plt.title('Number of Top 10 states in the dataset')
    plt.show()
    
    # word cloud business name
    top_words = Counter(df['name']).most_common(100)
    wordcloud = WordCloud(width=800, height=800, background_color='white').generate_from_frequencies(dict(top_words))

    plt.figure(figsize=(8,8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    
    # word cloud city
    top_words = Counter(df['city']).most_common(100)
    wordcloud = WordCloud(width=800, height=800, background_color='white').generate_from_frequencies(dict(top_words))
    plt.figure(figsize=(8,8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


# ### 3, Data cleaning

# In[5]:


def clean_data(df_left, df_right):
    # Drop NA values
    df_right['address'] = df_right['address'].fillna('')
    df_right['postal_code'] = df_right['postal_code'].fillna('')
    df_right['categories'] = df_right['categories'].fillna('')

    # Drop unnecessary columns
    df_left = df_left.drop('size', axis=1)
    df_right = df_right.drop('categories', axis=1)

    # Correct the format for postal_code column in df_right
    df_right['postal_code'] = df_right['postal_code'].astype(str)
    df_right['postal_code'] = df_right['postal_code'].str.slice(0, -2)

    df_right['postal_code'] = df_right['postal_code'].astype(str)
    df_left['zip_code'] = df_left['zip_code'].str.slice(0, -5)

    # Rename columns
    df_right = df_right.rename(columns={'postal_code': 'zip_code'})

    return df_left, df_right


# In[6]:


def preprocess(df):
    df['name'] = df['name'].fillna('')
    df['address'] = df['address'].fillna('')
    df['city'] = df['city'].fillna('')
    df['state'] = df['state'].fillna('')
    df['zip_code'] = df['zip_code'].fillna('')
    df['name'] = df['name'].str.lower().str.replace('[^\w\s]','')
    df['address'] = df['address'].str.lower().str.replace('[^\w\s]','')
    df['city'] = df['city'].str.lower().str.replace('[^\w\s]','')
    df['state'] = df['state'].str.lower().str.replace('[^\w\s]','')
    df['zip_code_tokens'] = df['zip_code'].str.split()
    df['name_tokens'] = df['name'].str.split()
    df['address_tokens'] = df['address'].str.split()
    df['city_tokens'] = df['city'].str.split()
    df['state_tokens'] = df['state'].str.split()
    df['blocking_key'] = df['name_tokens'].apply(lambda x: x[0][0] + x[1][:3] if len(x) > 1 else '')
    def shorten_rd(address):
        address = re.sub(r" street(?=$| [NE(So|S$)(We|W$)])", ' st', address)
        address = re.sub(r" road(?=$| [NE(So|S$)(We|W$)])", ' rd', address)
        address = re.sub(r"(?<!The) avenue(?=$| [NE(So|S$)(We|W$)])", ' ave', address)
        address = re.sub(r" close(?=$| [NE(So|S$)(We|W$)])", ' cl', address)
        address = re.sub(r" court(?=$| [NE(So|S$)(We|W$)])", ' ct', address)
        address = re.sub(r"(?<!The) crescent(?=$| [NE(So|S$)(We|W$)])", ' cres', address)
        address = re.sub(r" boulevarde?(?=$| [NE(So|S$)(We|W$)])", ' blvd', address)
        address = re.sub(r" drive(?=$| [NE(So|S$)(We|W$)])", ' dr', address)
        address = re.sub(r" lane(?=$| [NE(So|S$)(We|W$)])", ' ln', address)
        address = re.sub(r" place(?=$| [NE(So|S$)(We|W$)])", ' pl', address)
        address = re.sub(r" square(?=$| [NE(So|S$)(We|W$)])", ' sq', address)
        address = re.sub(r"(?<!The) parade(?=$| [NE(So|S$)(We|W$)])", ' pde', address)
        address = re.sub(r" circuit(?=$| [NE(So|S$)(We|W$)])", ' cct', address)
        address = re.sub(r"\b(north|south|east|west)\b", lambda m: m.group(1)[0], address)
        return address
    df['address'] = df['address'].map(shorten_rd)
    return df


# In[7]:


def block(df):
    block_dict = {}
    for index, row in df.iterrows():
        if row['blocking_key'] not in block_dict:
            block_dict[row['blocking_key']] = [index]
        else:
            block_dict[row['blocking_key']].append(index)
    return block_dict


# ### 4, Matching

# In[8]:


def matching(block_dict_left, block_dict_right, df_left, df_right):
    matches = []
    for key in block_dict_left:
        if key in block_dict_right:
            for left_idx in block_dict_left[key]:
                left_name = df_left.loc[left_idx, 'name']
                left_address = df_left.loc[left_idx, 'address']
                left_city = df_left.loc[left_idx, 'city']
                left_state = df_left.loc[left_idx, 'state']
                left_zip_code_tokens = df_left.loc[left_idx, 'zip_code_tokens']
                left_tokens = left_name.split() + left_address.split() + left_city.split() + left_state.split() + left_zip_code_tokens
                for right_idx in block_dict_right[key]:
                    right_name = df_right['name'][right_idx]
                    right_address = df_right['address'][right_idx]
                    right_city = df_right['city'][right_idx]
                    right_state = df_right['state'][right_idx]
                    right_zip_code_tokens = df_right['zip_code_tokens'][right_idx]
                    right_tokens = right_name.split() + right_address.split() + right_city.split() + right_state.split() + right_zip_code_tokens
                    name_sim = jellyfish.jaro_winkler(left_name, right_name)
                    address_sim = jellyfish.jaro_winkler(left_address, right_address)
                    city_sim = jellyfish.jaro_winkler(left_city, right_city)
                    state_sim = jellyfish.jaro_winkler(left_state, right_state)
                    token_sim = sum([max([jellyfish.jaro_winkler(left_token, right_token) for right_token in right_tokens]) for left_token in left_tokens]) / len(left_tokens)
                    sim_score = 0.4 * name_sim + 0.3 * address_sim + 0.1 * city_sim + 0.1 * state_sim + 0.1 * token_sim
                    if sim_score > 0.8:
                        match = (df_right['entity_id'][right_idx], df_left['business_id'][left_idx], sim_score)
                        matches.append(match)
    # output matches with score greater than 0.8
    matches = sorted(matches, key=lambda x: x[2], reverse=True)
    matches = [(x[0], x[1], round(x[2], 2)) for x in matches if x[2] > 0.93]
    return matches


# ### 5, CSV output

# In[ ]:


def csv_writer(matches, output_file):
    confidence_matches = pd.DataFrame(matches, columns=['entity_id', 'business_id', 'confidence_score'])
    confidence_matches.to_csv(output_file, index=False)

