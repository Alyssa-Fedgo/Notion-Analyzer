
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

# Load Notion Token securely
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
if not NOTION_TOKEN:
    raise ValueError("Missing NOTION_TOKEN environment variable.")

#connect to client
notion = Client(auth = NOTION_TOKEN)

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
    return pd.DataFrame(rows)
   
def classify_sections(df):
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

    return sections

def merge_sections(section_dfs):
    """  Input listing of sections
         Output: a merged dataframe so each row is an entry
         Purpose: usable dataset
    """
    return reduce(lambda left, right: pd.merge(left, right, on='page_id', how='outer'), section_dfs)

def format_output(df):
    """ Input is a merged dataframe
        Output is a formatted dataframe
        Purpose is to add more clarity to some columns
    """
    df['Grateful'] = "Today I'm grateful for " + df['Grateful']
    df['Intentions'] = "Today I intend to " + df['Intentions']
    df['Focus'] = "To make today a good day I would like to focus on " + df['Focus']
    df['Total'] = df['Grateful'] + " " + df['Intentions'] + " " + df['Focus'] + " " + df['Summary']
    return df

def main():
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
    formatted.to_csv('journal_output.csv', index=False)
    

if __name__ == "__main__":
    main()
