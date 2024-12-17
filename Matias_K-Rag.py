#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Breaks File into Chunks

def chunk_text(text, chunk_size=1000):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# Read the book in smaller chunks to avoid segmentation fault
def read_file_in_chunks(file_path, chunk_size=1000):
    with open(file_path, "r", encoding="utf-8") as file:
        while True:
            chunk = file.read(chunk_size * 6)  # Read approximately 1000 words (assuming average word length of 6 characters)
            if not chunk:
                break
            yield chunk

# Process the file in chunks
file_path = "./cleaned_GOT.txt"
chunks = list(read_file_in_chunks(file_path))
len(chunks)


# In[2]:


import spacy
import networkx as nx
import matplotlib.pyplot as plt

# In[4]:


# Copy pasted from instructions

def extract_relationships(doc):
    relationships = []
    for sent in doc.sents:
        root = sent.root
        subject = None
        obj = None
        for child in root.children:
            if child.dep_ == "nsubj":
                subject = child
            if child.dep_ in ["dobj", "pobj"]:
                obj = child
        if subject and obj:
            relationships.append((subject, root, obj))
    return relationships

def process_document(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    relationships = extract_relationships(doc)
    return entities, relationships


# In[5]:


def build_knowledge_graph(documents):
    G = nx.DiGraph()
    for doc in documents:
        entities, relationships = process_document(doc)
        for entity, entity_type in entities:
            G.add_node(entity, type=entity_type)
        for subj, pred, obj in relationships:
            G.add_edge(subj.text, obj.text, relation=pred.text)
    return G

# Loads spacy model since we were having issues making a RAG model
print("Loading up spacy model...")
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('sentencizer')
documents = chunks  # All chunks of the book


# In[10]:

from langchain import PromptTemplate, LLMChain
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
from dotenv import load_dotenv
import tiktoken

print("Loading up RAG model...")

# Load environment variables from the .env file
load_dotenv()

# Get key from the env
openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialize components
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = FAISS.from_texts(documents, embeddings)
llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

# RAG prompt template
rag_template = """Context: {context}

Question: {question}

Answer:"""
rag_prompt = PromptTemplate(template=rag_template, input_variables=["context", "question"])
rag_chain = LLMChain(llm=llm, prompt=rag_prompt)

def rag_query(question, k=3):
    relevant_docs = vectorstore.similarity_search(question, k=k)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    # Truncate context if it exceeds the maximum token limit
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    context_tokens = encoding.encode(context)
    question_tokens = encoding.encode(question)

    max_tokens = 4097 - 256 - 10 # Reserve 256 tokens for the completion and an additional 10 that I think is coming from the template but unsure
    max_tokens_minus_question = max_tokens - len(question_tokens)
    
    if len(context_tokens) > max_tokens_minus_question:
        context_tokens = context_tokens[:max_tokens_minus_question]
        context = encoding.decode(context_tokens)

    print("Number of context tokens: " + str(len(context_tokens)) + ", number of question tokens: " + str(len(question_tokens)))

    return rag_chain.run(context=context, question=question)


# In[12]:

print("Loading up KRAG augmentation...")

def get_relevant_triples(question, graph, k=5):
    entities = nlp(question).ents
    relevant_triples = []
    for entity in entities:
        if entity.text in graph:
            neighbors = list(graph.neighbors(entity.text))
            for neighbor in neighbors[:k]:
                edge_data = graph.get_edge_data(entity.text, neighbor)
                relevant_triples.append(f"{entity.text} {edge_data['relation']} {neighbor}")
    return relevant_triples

def krag_query(question, knowledge_graph, k=3):
    relevant_docs = vectorstore.similarity_search(question, k=k)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    relevant_triples = get_relevant_triples(question, knowledge_graph)
    triples_context = "\n".join(relevant_triples)
    
    krag_template = """Context: {context}

Relevant Knowledge Graph Triples:
{triples_context}

Question: {question}

Answer:"""
    krag_prompt = PromptTemplate(template=krag_template, input_variables=["context", "triples_context", "question"])
    krag_chain = LLMChain(llm=llm, prompt=krag_prompt)
    
    # Truncate context if it exceeds the maximum token limit
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    context_tokens = encoding.encode(context)
    question_tokens = encoding.encode(question)
    triples_context_tokens = encoding.encode(triples_context)

    max_tokens = 4097 - 256 - 30 # Reserve 256 tokens for the completion and an additional 10 that I think is coming from the template but unsure
    max_tokens_minus_question_and_triples_tokens = max_tokens - len(question_tokens) - len(triples_context_tokens)
    
    if len(context_tokens) > max_tokens_minus_question_and_triples_tokens:
        context_tokens = context_tokens[:max_tokens_minus_question_and_triples_tokens]
        context = encoding.decode(context_tokens)

    print("Number of context tokens: " + str(len(context_tokens)) + ", number of question tokens: " + str(len(question_tokens)) + ", number of triple tokens: " + str(len(question_tokens)))

    
    return krag_chain.run(context=context, triples_context=triples_context, question=question)

import os

def main():
    print("Program Start")
    knowledge_graph_input = input("Make knowledge graph? (y/n) ")
    knowledge_graph = None
    
    if(knowledge_graph_input.lower() == "y"):
        print("Making knowledge graph...")
        knowledge_graph = build_knowledge_graph(documents)
    else:
        print("K-rag declined.")
    
    # Check the current directory for the number of the log file
    log_number = 1
    while os.path.exists(f"response_log_{log_number}.txt"):
        log_number += 1
    
    # Open the log file in append mode
    log_file = open(f"response_log_{log_number}.txt", "a")
    
    experiment_mode = input("Experiment Mode? (y/n) ")
    
    if(experiment_mode.lower() == 'y'):
        method = 'both'
    
    # Main loop of the program
    while True:
        question = input("Enter your question (or 'q' to quit): ")
        if (question == 'q'):
            break
        
        if (experiment_mode.lower() != 'y'):
            method = input("Enter 'krag' to use Krag, 'rag' to use Rag, or 'both' to get both answers: ")
        
        if (method == 'krag'):
            answer = krag_query(question, knowledge_graph)
            print("Krag Answer:", answer)
            log_file.write(f"Question: {question}\n")
            log_file.write(f"Krag Answer: {answer}\n")
        elif (method == 'rag'):
            answer = rag_query(question)
            print("Rag Answer:", answer)
            log_file.write(f"Question: {question}\n")
            log_file.write(f"Rag Answer: {answer}\n")
        elif (method == 'both'):
            krag_answer = krag_query(question, knowledge_graph)
            rag_answer = rag_query(question)
            print("Krag Answer:", krag_answer)
            print("Rag Answer:", rag_answer)
            log_file.write(f"Question: {question}\n")
            log_file.write(f"Krag Answer: {krag_answer}\n")
            log_file.write(f"Rag Answer: {rag_answer}\n")
        else:
            print("Invalid method selected.")
            continue
    
    # Close the log file
    log_file.close()

if __name__ == "__main__":
    main()