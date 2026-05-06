import pandas as pd
import requests
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

df = pd.read_csv("all_image_urls.csv")

os.makedirs("images", exist_ok=True)

def download(data):
    i, row = data
    try:
        url = row.iloc[1] 
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            with open(f"images/{i}.jpg", "wb") as f:
                f.write(response.content)
    except:
        pass

with ThreadPoolExecutor(max_workers=20) as executor:
    list(tqdm(executor.map(download, df.iterrows()), total=len(df)))