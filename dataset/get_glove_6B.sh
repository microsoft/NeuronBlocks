preprocess_exec="sed -f tokenizer.sed"

glovepath='http://nlp.stanford.edu/data/glove.6B.zip'

ZIPTOOL="unzip"

# GloVe
echo $glovepath
if [ ! -d "/GloVe/"];then
    mkdir GloVe
fi
curl -LO $glovepath
$ZIPTOOL glove.6B.zip -d GloVe/
rm glove.6B.zip

