import openai
import pandas as pd
from time import sleep

# TODO environment variable!
openai.organization = "org-j7UX486JsMOhacdgXJwYJwSc"
# get this from top-right dropdown on OpenAI under organization > settings
openai.api_key = "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"
# get API key from top-right dropdown on OpenAI website

openai.Engine.list()  # check we have authenticated

MODEL = "text-similarity-ada-002"  # check if correct

df_b = pd.read_csv('/Users/misha/Desktop/Bachelor-Thesis/BA/data_sets/the_one/big.csv')

# 150,000 tokens per request
# 20 requests per minute
# First read the max tokens per review + max tokens per title and the 150k / that amount = how many requests to be made
# max by torchtext is 203 title is 26 lets say 230 (one is buffer then)

# TODO: There is a more efficient way to write following code, improve!

reqs = []
total = df_b['Description'].tolist()
for i in range(6):
    print(i)
    m = i * 700
    res = total[m:m + 700]
    print(len(res))
    reqs.append(res)

results_b = []
for i in reqs:
    res = openai.Embedding.create(input=i, engine=MODEL)
    results_b.append(res)
    sleep(60)

embeds = []
for r in results_b:
    embeds.append([record['embedding'] for record in r['data']])

em = []

for e in embeds:
    for i in e:
        em.append(i)

df_b['Embedding'] = em

df_b.to_csv('/Users/misha/Desktop/Bachelor-Thesis/BA/data_sets/the_one/big_e.csv', sep=',', index=False)

