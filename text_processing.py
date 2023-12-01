import re
import requests
import pandas as pd
import transformers
from transformers import LongformerTokenizer
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")


def fetch_and_save_wiki_text(title):
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
        },
    ).json()

    page = next(iter(response["query"]["pages"].values()))
    wiki_text = page["extract"]

    return wiki_text

def clean_text(text):
    # Remove special characters except "."
    text = re.sub(r'[^A-Za-z0-9\s.\(\)\[\]\{\}]+', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def count_tokens(text):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    return len(tokens)

def process_wonders_data():
    wonders_cities = [
        'Beirut',
        'Doha',
        'Durban',
        'Havana',
        'La Paz',
        'Vigan',
    ]

    data = []
    for wonder_city in wonders_cities:
        info = fetch_and_save_wiki_text(wonder_city)
        tokens = tokenizer.encode(info, add_special_tokens=True, truncation=True, max_length=30000)
        num_tokens = len(tokens)
        data.append([wonder_city, info, num_tokens])

    df = pd.DataFrame(data, columns=["wonder_city", "information", "num_tokens"])
    df["cleaned_information"] = df["information"].apply(clean_text)
    df["token_count"] = df["cleaned_information"].apply(count_tokens)
    beirut_data = df[df["wonder_city"] == "Beirut"].copy()
    return beirut_data
