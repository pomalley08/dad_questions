import numpy as np
import pandas as pd
import tensorflow_hub as hub


embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

titles = pd.read_csv("all_titles.csv", encoding="utf-8", dtype={"titles":"string"}).dropna()

docs = titles["titles"].tolist()

embeddings = embed(docs)

np.savetxt('embeddings.csv', embeddings, delimiter=",")
