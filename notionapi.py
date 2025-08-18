
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
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
from scipy.stats import pearsonr
from datetime import date


# Download stopwords
def safe_nltk_download(pkg: str):
    """ Checks if nltk library is already downloaded. If not, then download.
    """
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg.split("/")[-1])

safe_nltk_download("corpora/stopwords")
safe_nltk_download("tokenizers/punkt")
safe_nltk_download("sentiment/vader_lexicon")
STOPWORDS = set(nltk.corpus.stopwords.words("english"))


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
   
def classify_sections(df:pd.DataFrame)->pd.DataFrame:
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

        if 'learn' in category:
            content_df.rename(columns={'value': 'Summary'}, inplace=True)
        elif 'grateful' in category:
            content_df.rename(columns={'value': 'Grateful'}, inplace=True)
        elif 'intention' in category:
            content_df.rename(columns={'value': 'Intentions'}, inplace=True)
        elif 'day' in category:
            content_df.rename(columns={'value': 'Focus'}, inplace=True)
        else:
            continue

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
    df['Total'] = df['Grateful'] + " " + df['Intentions'] + " " + df['Focus'] + " " + df['Summary']
    print('Cleaned up columns')
    return df

def process_words(word_list:list)->list:
    """ Input: list of words
        Output: cleaned up list of words
        Purpose: lowercase a column and remove stop words
    """
    
    return [str(word).lower() for word in word_list if ((str(word).isalpha()) & (str(word).lower() not in STOPWORDS))]


def nlp_prep(df:pd.DataFrame)->pd.DataFrame:
    """ Input: df
        Output: df
        Purpose: create a column with tokenized and cleaned column for nlp
    """
    # Apply word_tokenize to the 'text' column
    df['tokenized'] = df['Total'].astype(str).apply(word_tokenize)
    df['tokenized'] = df['tokenized'].apply(process_words)
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
    df['polarity_cat'] = np.where(df['polarity'] >0, 'Positive', np.where(df['polarity'] ==0,'NA','Negative'))
    print('Return polarity')
    return df

def correlation_on_wordfreq(df:pd.DataFrame)->pd.DataFrame:
    results = []
    # Step 1 — Get unique words 
    all_words = sorted(set(word for tokens in df['word_list'] for word in tokens)) 
    all_words = list(set(all_words))

    # Step 2 — Create binary indicator columns 
    for word in all_words: 
        df[word] = df['word_list'].apply(lambda tokens: int(word in tokens))

    for word in set(word for tokens in df['word_list'] for word in tokens):
        try:
            r, p = pearsonr(df[word], df['polarity'])
            results.append({"word": word, "correlation": r, "p_value": p})
        except Exception:
            continue

    sig_results = (
        pd.DataFrame(results)
        .query("p_value < 0.05")
        .sort_values("correlation", key=lambda x: abs(x), ascending=False)
    )
    return sig_results

def summary(df:pd.DataFrame)->str:
    """find frequency of negative days"""
    results = df.groupby('polarity_cat').size().reset_index(name='count')
    neg = results.loc[results['polarity_cat'] == 'Negative', 'count'].iloc[0]
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
        block_df.to_csv('testing.csv')
        sections = classify_sections(block_df)
        if not sections:
            print("No journal sections found.")

        merged = merge_sections(sections)
        merged.to_csv('merged.csv')
        formatted = format_output(merged)
        nlp_prepped = nlp_prep(formatted)
        word_freq = word_frequency(nlp_prepped)
        polarity_score = polarity(word_freq)
        polarity_score.to_csv('journal_output.csv', index=False)
        neg_words = polarity_score[polarity_score['polarity']<0][['word_frequency']]
        neg_words_txt = neg_words.to_string()
        print('Output to CSV')
        corr_df = correlation_on_wordfreq(polarity_score)
        corr_df.to_csv('journal_corr.csv', index=False)
        corr_txt = corr_df.to_string()
        print('Output to CSV')
        freq = summary(polarity_score)
        add_page_to_database('Results of NLP', freq, corr_txt, neg_words_txt)
    except Exception as e:
        print(f'{e}')
    

    

if __name__ == "__main__":
    main()
