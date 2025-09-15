##############################################################################
### Program:     notionapi.py
### Developer:   Alyssa Fedgo
### Date:        July 2025
### Version:     1.0
### Description: PUll Notion documents to understand what impacts by mood
### Changes:     1.0 Initial
##############################################################################

import os
from notion_client import Client
import requests 
import pandas as pd
from functools import reduce
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
import spacy
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
from scipy.stats import pearsonr
from datetime import date
import matplotlib.pyplot as plt

nltk_data_dir = "/opt/airflow/nltk_data"
os.makedirs(nltk_data_dir, exist_ok=True)

# Make sure Airflow user can write
os.chmod(nltk_data_dir, 0o777)
nltk.data.path.append(nltk_data_dir)

load_dotenv()

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Download stopwords
def safe_nltk_download(pkg: str):
    """ Checks if nltk library is already downloaded. If not, then download.
    """
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg.split("/")[-1], download_dir=nltk_data_dir)

safe_nltk_download("corpora/stopwords")
safe_nltk_download("tokenizers/punkt_tab")
safe_nltk_download("sentiment/vader_lexicon")
safe_nltk_download("vader_lexicon")
safe_nltk_download("wordnet")
safe_nltk_download("omw-1.4")
safe_nltk_download("averaged_perceptron_tagger_eng")


STOPWORDS = set(nltk.corpus.stopwords.words("english"))


lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()
print('Downloaded stopwords')

# Load Notion Token securely
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_TOKEN2 = os.getenv("NOTION_TOKEN2")
DATABASE_ID = os.getenv("DATABASE_ID")
if not NOTION_TOKEN:
    raise ValueError("Missing NOTION_TOKEN environment variable.")
if not NOTION_TOKEN2:
    raise ValueError("missing NOTION_TOKEN2 environment variable.")
if not DATABASE_ID:
    raise ValueError("Missing DATABASE_ID environment variable")
print('Retrieved notion token')

headers = {
    "Authorization": f"Bearer {NOTION_TOKEN2}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28"  # latest stable API version
}

#connect to client
notion = Client(auth = NOTION_TOKEN)
notion2 = Client(auth = NOTION_TOKEN2)
print('Connected to client')

def get_all_page_ids(parent_id=None)->list:
    """
         Inputs: parent_id. Default value is None. Intended data type is string.
         Output: List of page_ids
         Purpose: Get pages from journal
    """
    page_ids = []
    # If parent_id is None, use the search endpoint to fetch top-level pages
    if parent_id is None:
        response = notion.search(filter={"property": "object", "value": "page"})
        for result in response.get("results", []):
            page_ids.append(result["id"])
            # Recursively fetch child pages
            page_ids.extend(get_all_page_ids(result["id"]))
    else:
        # Fetch child blocks of the parent page
        response = notion.blocks.children.list(block_id=parent_id)
        for block in response.get("results", []):
            if block["type"] == "child_page":
                page_ids.append(block["id"])
                # Recursively fetch child pages
                page_ids.extend(get_all_page_ids(block["id"]))
    print('Returned page ids')
    return page_ids

def extract_blocks(PAGE_IDS:list)->list:
    """
        Input: PAGE_IDS which is a list. This will be output from the method get_all_page_ids
        Output: List of blocks on each page
        Purpose: retrieve text on each journal page
    """
    rows = []
    for page_id in PAGE_IDS:
        response = notion.blocks.children.list(block_id=page_id)
        blocks = response.get('results', [])
        section_index = 0

        for block in blocks:
            block_type = block.get("type")
            if block_type in ['unsupported', 'child_database']:
                continue

            content = block.get(block_type, {}).get("rich_text", [])
            text = content[0]["plain_text"] if content else None

            if "heading" in block_type:
                section_index += 1

            rows.append({
                "id": block.get("id"),
                "page_id": page_id,
                "created_time": block.get("created_time"),
                "row": section_index,
                "type": block_type,
                "value": text
            })
    print('Returned blocks')
    return pd.DataFrame(rows)
   
def classify_sections(df:pd.DataFrame)->list:
    """ Input: Dataframe
        Output: List
        Purpose: Categorize each row as having information for either summary, grateful, focus, or intenions
    """
    sections = []
    for row_id in df['row'].unique():
        section = df[(df['row'] == row_id) & (df['value'].notna()) ]
        if section.empty: continue
        category = section.iloc[0]['value'].lower()
        content_df = section[~section['type'].str.contains('heading')].groupby('page_id')['value'].agg(', '.join).reset_index()

        if 'grateful' in category:
            content_df.rename(columns={'value': 'Grateful'}, inplace=True)
        elif 'intention' in category:
            content_df.rename(columns={'value': 'Intentions'}, inplace=True)
        elif 'day' in category:
            content_df.rename(columns={'value': 'Focus'}, inplace=True)
        else:
            content_df.rename(columns={'value': 'Summary'}, inplace=True)
           # continue

        sections.append(content_df)
    print('Clasiified row types')
    return sections

def merge_sections(section_dfs:pd.DataFrame)->pd.DataFrame:
    """  Input listing of sections
         Output: a merged dataframe so each row is an entry
         Purpose: usable dataset
    """
    print('Merged journal entries to form one row')
    return reduce(lambda left, right: pd.merge(left, right, on='page_id', how='outer'), section_dfs)

def format_output(df:pd.DataFrame)->pd.DataFrame:
    """ Input is a merged dataframe
        Output is a formatted dataframe
        Purpose is to add more clarity to some columns
    """
    df['Grateful'] = "Today I'm grateful for " + df['Grateful']
    df['Intentions'] = "Today I intend to " + df['Intentions']
    df['Focus'] = "To make today a good day I would like to focus on " + df['Focus']
    df['Total'] = df['Grateful'] + ". " + df['Intentions'] + ". " + df['Focus'] + ". " + df['Summary']
    print('Cleaned up columns')
    return df


def get_wordnet_pos(tag):
    """ Map NLTK POS tags to WordNet POS tags"""
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def process_words(word_list:list)->list:
    """ Input: list of words
        Output: cleaned up list of words
        Purpose: lowercase a column and remove stop words
    """
    
    return [str(word).lower() for word in word_list if ((str(word).isalpha()) & (str(word).lower() not in STOPWORDS))]

def lemma_words(word_list:list)-> list:
    pos_tags = pos_tag(word_list)

    # Lemmatize with POS tags
    lemmatized_sentence = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags
    ]
    return lemmatized_sentence

def nlp_prep(df:pd.DataFrame)->pd.DataFrame:
    """ Input: df
        Output: df
        Purpose: create a column with tokenized and cleaned column for nlp
    """
    # Apply word_tokenize to the 'text' column
    df['tokenized'] = df['Total'].astype(str).apply(word_tokenize)
    df['tokenized'] = df['tokenized'].apply(process_words)
    df['tokenized'] = df['tokenized'].apply(lemma_words)
    print('Completed nlp prep')
    return df

def word_frequency(df:pd.DataFrame)->pd.DataFrame:
    """ Input: dataframe
        Output: datafarame
        Purpose: return top 10 frequent words in each row
    """
    df['word_frequency'] = df.tokenized.apply(lambda tokens: nltk.FreqDist(tokens).most_common(10))
    df['word_list'] = df.word_frequency.apply(lambda x: [t[0] for t in x] )
    print('Applied word frequency')
    return df

def polarity(df:pd.DataFrame)->pd.DataFrame:
    """Input: dataframe
       Output: dataframe
       purpose: Determine if journal entry is overall positive or negative
    """
    df['polarity'] = df['tokenized'].apply(lambda tokens: sia.polarity_scores(" ".join(tokens))['compound'])
    df['polarity_cat'] = np.where(df['polarity'] >0.5, 'Positive', 'Negative')
    print('Return polarity')
    return df

def get_top_contributors(tokens, top_n=10):
    """Get the top 10 words per polarity score"""
    # Score each token individually
    token_scores = [(word, sia.polarity_scores(word)['compound']) for word in tokens]
    # Sort by absolute score
    token_scores = sorted(token_scores, key=lambda x: abs(x[1]), reverse=True)
    # Keep only top N with non-zero contribution
    top_tokens = [(w, s) for w, s in token_scores if abs(s) >=.25]
    #top_tokens = [(w, s) for w, s in token_scores if s != 0][:top_n]
    return top_tokens

def find_subjects(text, contributors):

    # Handle missing or non-string text
    if not isinstance(text, str) or not text.strip():
        return [] 
    
    doc = nlp(text)
    results = []

    for word, score in contributors:
        # Find all tokens matching contributor word
        matches = [t for t in doc if t.text.lower() == word.lower()]
        subjects = []

        for token in matches:
            # 1️ Direct nominal subjects (nsubj)
            subjects.extend([child.text for child in token.children if child.dep_ in ("nsubj", "nsubjpass")])

            # 2️ Prepositional objects (for adjectives like "grateful for X")
            for child in token.children:
                if child.dep_ == "prep":  # preposition
                    subjects.extend([grandchild.text for grandchild in child.children if grandchild.dep_ == "pobj"])

            # 3️ Direct objects (dobj) or object of xcomp
            subjects.extend([child.text for child in token.children if child.dep_ in ("dobj", "xcomp")])

            # 4 Climb to head to see if it has prepositional objects
            head = token.head
            if head != token:
                for child in head.children:
                    if child.dep_ == "prep":
                        subjects.extend([gc.text for gc in child.children if gc.dep_ == "pobj"])

        # Deduplicate subjects and fallback to "I" if none
        subjects_clean = list(set(subjects)) if subjects else ["I"]

        results.append({
            "word": word,
            "score": score,
            "subjects": subjects_clean
        })

    return results
    

def output_related_words(df:pd.DataFrame):
    related_words= df['contributors_with_related'].tolist()
    words = {'positive':set(),
             'negative':set()}
    for l in related_words:
        if l != []:
            for d in l:
                if d['score']>0.35:
                    direction = 'positive'
                else:
                    direction='negative'
                for noun in d['subjects']:
                    if noun.lower() not in ['get', 'myself', 'me', 'he', 'that', 'you', 'what', 'something',
                                            'it', 'habit', 'day', 'her', 'someone', 'take', 'them', 'letting',
                                            'be', 'who', 'do', 'getting', 'average', 'i', 'see', 'have',
                                            'stuff', 'this', 'any', 'morning', 'us', 'gateway', 'watching',
                                            'nights', 'night', 'clause', 'caring', 'post', 'days', 'make','pace',
                                            'speak', 'start']:
                        if direction == 'positive':
                            words['positive'].add(noun)
                        else:
                            words['negative'].add(noun)

    return words

def summary(df:pd.DataFrame)->str:
    """find frequency of negative days"""
    results = df.groupby('polarity_cat').size().reset_index(name='count')
    neg = results.loc[results['polarity_cat'] == 'Negative', 'count'].sum()
    total = len(df)
    perc = (neg/total)*100
    return f'{neg} out of {total} days were negative. This means {round(perc,2)}% of days were negative'

def add_page_to_database(title: str, mood: str, content:str, tags:str):
    create_url = "https://api.notion.com/v1/pages"

    # Current date and time
    now = date.today()
    formatted_date = now.strftime("%B %d, %Y")

    new_page_data = {
        "parent": { "database_id": DATABASE_ID },
        "properties": {
            "Title": {
                "title": [
                    { "text": { "content": title } }
                ]
            },
            "Date": {
                "rich_text": [
                    { "text": { "content": formatted_date } }
                ]
            },
            "Mood": {
                "rich_text": [
                    { "text": { "content": mood} }
                ]
            },
            "Tags": {
                "rich_text": [
                    { "text": { "content": tags } }
                ]
            },
            "Content": {
                "rich_text": [
                    { "text": { "content": content } }
                ]
            }
        }
    }

    response = requests.post(create_url, headers=headers, json=new_page_data)
    
    if response.status_code == 200:
        print("Page created successfully!")
    else:
        print("Failed to create page:", response.status_code, response.text)


def main():
    try:
        # Fetch all page IDs
        all_page_ids = get_all_page_ids()
        #retreive text on journal entries
        block_df = extract_blocks(all_page_ids)
        sections = classify_sections(block_df)
        if not sections:
            print("No journal sections found.")

        merged = merge_sections(sections)
        formatted = format_output(merged)
        nlp_prepped = nlp_prep(formatted)
        word_freq = word_frequency(nlp_prepped)
        polarity_score = polarity(word_freq)
        # Apply function to dataframe
        polarity_score["top_contributors"] = polarity_score["tokenized"].apply(lambda x: get_top_contributors(x, top_n=10))

        polarity_score["contributors_with_related"] = polarity_score.apply(
            lambda row: find_subjects(row["Total"], row["top_contributors"]),
            axis=1
        )
        polarity_score.to_csv('journal_output.csv', index=False)
        print('Output to CSV')


        plt.hist(polarity_score['polarity'], bins=30, edgecolor='black')
        plt.xlabel('Polarity Score')
        plt.ylabel('Frequency')
        plt.title('Histogram of Polarity Scores')
        plt.grid(axis='y', alpha=0.75)
        plt.show()


        related_words = output_related_words(polarity_score)
        positive = f'Positive factors in my life include {", ".join(related_words["positive"])}'
        negative = f'Negative factors in my life include {", ".join(related_words["negative"])}'
        freq = summary(polarity_score)
        add_page_to_database('Results of NLP', freq, positive, negative)
    except Exception as e:
        print(f'{e}')
    

    

if __name__ == "__main__":
    main()
