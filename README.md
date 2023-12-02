### Introduction
Simple GPT built using Chainlit for the chatbot. Data is scraped on confluence and stored as a Chroma vector_db

### Getting started
- Pip install all required dependencies.
- Run scraper.py to scrape data from confluence

### How to run
After data is scrapped and stored in Chroma DB, run `chainlit run ./frontend/main.py -w` to get chatbot up and running