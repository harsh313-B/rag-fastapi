# RAG FastAPI

A small Retrieval-Augmented Generation (RAG) application built with
Python, FastAPI, Hugging Face Transformers, Sentence Transformers,
and FAISS.

## Overview

This project demonstrates how a basic RAG system can retrieve relevant
information from a local knowledge document and use a small language
model to generate an answer.

The RAG pipeline is exposed through a FastAPI REST API.

## How It Works

```text
User Question
      ↓
Sentence Transformer
      ↓
Query Embedding
      ↓
FAISS Similarity Search
      ↓
Relevant Chunks
      ↓
Prompt + Retrieved Context
      ↓
FLAN-T5 Small
      ↓
Generated Answer
      ↓
FastAPI JSON Response