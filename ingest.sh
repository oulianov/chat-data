# Bash script to ingest data
# This involves scraping the data from the web and then cleaning up and putting in Weaviate.
# Error if any command fails
set -e
wget -r -A.html http://karpathy.github.io/2015/05/21/rnn-effectiveness/
python3 ingest.py
