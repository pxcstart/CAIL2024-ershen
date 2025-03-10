python -m pyserini.search.lucene \
  --index /home/peter/Challenge/baseline/data/index \
  --topics /home/peter/Challenge/baseline/data/clean_queries.tsv \
  --output /home/peter/Challenge/baseline/data/bm25_output.tsv \
  --bm25 \
  --hits 5 \
  --threads 1 \
  --batch-size 10 \
  --stopwords /home/peter/Challenge/baseline/data/others/stopword.txt
 




