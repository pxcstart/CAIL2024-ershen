python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input /home/peter/Challenge/baseline/data/input \
  --index /home/peter/Challenge/baseline/data/index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw \
  --stopwords /home/peter/Challenge/baseline/data/others/stopword.txt