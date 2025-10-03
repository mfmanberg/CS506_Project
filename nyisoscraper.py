import requests
from bs4 import BeautifulSoup
import os
import zipfile
import io
import pandas as pd
from tqdm import tqdm
from urllib.parse import urljoin

# URL of NYISO archived files page
url = "https://mis.nyiso.com/public/P-58Blist.htm"

# fetch page
r = requests.get(url)
r.raise_for_status()
soup = BeautifulSoup(r.text, "html.parser")

# folder for partitioned parquet dataset
dataset_dir = "nyiso_dataset"
os.makedirs(dataset_dir, exist_ok=True)

# find the single table
table = soup.find("table")

# collect all archive links
links = []
collect = False
for tr in table.find_all("tr"):
    text = tr.get_text(strip=True)
    if "Archived Files" in text:
        collect = True
        continue
    if collect:
        link = tr.find("a")
        if link and link.get("href"):
            file_url = urljoin(url, link["href"])  # safer join
            links.append(file_url)

# download + convert directly to partitioned Parquet
for file_url in tqdm(links, desc="Downloading & converting"):
    file_name = os.path.basename(file_url)

    # download into memory
    resp = requests.get(file_url, stream=True)
    resp.raise_for_status()

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            for member in zf.namelist():
                if member.endswith(".csv"):
                    with zf.open(member) as f:
                        df = pd.read_csv(f)

                        # Ensure we have a datetime column
                        # (You may need to adjust column name depending on NYISO format)
                        for col in df.columns:
                            if "time" in col.lower():
                                df[col] = pd.to_datetime(df[col], errors="coerce")
                                time_col = col
                                break
                        else:
                            print(f"No timestamp column in {member}, skipping.")
                            continue

                        # Add year/month columns
                        df["year"] = df[time_col].dt.year
                        df["month"] = df[time_col].dt.month.astype(str).str.zfill(2)

                        # Write into partitioned dataset
                        df.to_parquet(
                            dataset_dir,
                            engine="pyarrow",
                            compression="snappy",
                            partition_cols=["year", "month"],
                            index=False
                        )

    except zipfile.BadZipFile:
        print(f"Skipping {file_name}, not a valid ZIP.")

print(f"Done! Partitioned Parquet dataset created in: {dataset_dir}/")
