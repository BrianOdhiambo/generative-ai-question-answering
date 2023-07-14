from datasets import load_dataset
from tqdm.auto import tqdm # progress bar
import pinecone
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import BartTokenizer, BartForConditionalGeneration

from decouple import config

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load the dataset from huggingface in streaming mode and shuffle it
wiki_data = load_dataset(
    'vblagoje/wikipedia_snippets_streamed',
    split='train',
    streaming=True
).shuffle(seed=960)

# print(next(iter(wiki_data)))

history = wiki_data.filter(
    lambda d: d['section_title'].startswith('History')
)

total_doc_count = 50000

counter = 0
docs = []

# iterate through the dataset and apply our filter
for d in tqdm(history, total=total_doc_count):
    # extract the fields we need
    doc = {
        "article_title": d["article_title"],
        "section_title": d["section_title"],
        "passage_text": d["passage_text"]
    }

    # add the dict containing fields we need to docs list
    docs.append(doc)

    # stop iteration once we reach 50k
    if counter == total_doc_count:
        break

    # increase the counter on every iteration
    counter += 1

df = pd.DataFrame(docs)

# Initialize Pinecone index
pinecone.init(
    api_key=config('api_key'),
    environment=config('environment')
)

index_name = "abstractive-question-answering"

# check if the abstractive-question-answering index exists
if index_name not in pinecone.list_indexes():
    #create the index if it does not exist
    pinecone.create_index(
        index_name,
        dimension=768,
        metric='cosine'
    )

# connect to abstractive-question-answering index we created
index = pinecone.Index(index_name)

# Initializer Retriever

# load the retriever model from huggingface model hub
retriever = SentenceTransformer(
    "flax-sentence-embeddings/all_datasets_v3_mpnet-base",
    device=device
)

# Generate embeddings and upsert

# we will use batches of 64
batch_size = 64

for i in tqdm(range(0, len(df), batch_size)):
  # find end of batch
  i_end = min(i+batch_size, len(df))

  # extract batch
  batch = df.iloc[i:i_end]

  # generate embeddings for batch
  emb = retriever.encode(batch["passage_text"].tolist()).tolist()

  # get metadata
  meta = batch.to_dict(orient="records")

  # create unique IDs
  ids = [f"{idx}" for idx in range(i, i_end)]

  # add all upsert list
  to_upsert = list(zip(ids, emb, meta))

  # upsert/insert these records to pinecone
  _ = index.upsert(vectors=to_upsert)

# check that we have all vectors in index
index.describe_index_stats()

# Initialize Generator
# load bart tokenizer and model from huggingface
tokenizer = BartTokenizer.from_pretrained("vblagoje/bart_lfqa")
generator = BartForConditionalGeneration.from_pretrained("vblagoje/bart_lfqa")

def query_pinecone(query, top_k):
  # generate embeddings for the query
  xq = retriever.encode([query]).tolist()

  # search pinecone index for context passage with the answer
  xc = index.query(xq, top_k=top_k, include_metadata=True)
  return xc

def format_query(query, context):
  # extract passage_text from Pinecone search results and add the <P> tag
  context = [f"<P> {m['metadata']['passage_text']}" for m in context]

  # concatinate all context passages
  context = " ".join(context)

  # concatenate the query and context passages
  query = f"question: {query} context: {context}"
  return query

# result = query_pinecone(query, top_k=1)
# format the query in the form generator expects the input
# query = format_query(query, result["matches"])
# print(query)

def generate_answer(query):
  # tokenize the query to get input_ids
  inputs = tokenizer([query], max_length=1024, return_tensors="pt")

  # use generator to predict output ids
  ids = generator.generate(inputs["input_ids"], num_beams=2, min_length=20, max_length=40)

  # use tokenizer to decode the output ids
  answer = tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
  return print(answer)

print("Ask question: \n")
query = str(input())
generate_answer(query)