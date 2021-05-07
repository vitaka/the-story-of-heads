#! /bin/bash
MYFULLPATH=$(readlink -f $0)
CURDIR=$(dirname $MYFULLPATH)

DEVPREFIX="alldevtest.bpe"
DEVFILTERED="alldevtest.bpe.samesize"

moses_scripts=/neural/mtl-emnlp/submodules/moses-scripts/scripts/
nomalizer=$moses_scripts/tokenizer/normalize-punctuation.perl
tokenizer=$moses_scripts/tokenizer/tokenizer.perl
detokenizer=$moses_scripts/tokenizer/detokenizer.perl
clean_corpus=$moses_scripts/training/clean-corpus-n.perl
train_truecaser=$moses_scripts/recaser/train-truecaser.perl
truecaser=$moses_scripts/recaser/truecase.perl
detruecaser=$moses_scripts/recaser/detruecase.perl

SL=$1
TL=$2
TRAINDIR=$3
DATADIR=$4

CORPUSDIR="$TRAINDIR/corpus"


#Tokenize, truecase, BPE
for L in $SL $TL ; do
  cat $DATADIR/alldevtest.$L  | $nomalizer -l $L | $tokenizer -a -no-escape -l $L | $truecaser -model $TRAINDIR/model/truecaser/truecase-model.$L | subword-nmt apply-bpe --vocabulary $TRAINDIR/model/vocab.$SL$TL.bpe.bpevocab.$L --vocabulary-threshold 1 -c $TRAINDIR/model/vocab.$SL$TL.bpe >$CORPUSDIR/$DEVPREFIX.$L
done

#Count tokens
for L in $SL $TL ; do
    cat $CORPUSDIR/$DEVPREFIX.$L | awk -F ' ' '{print NF}' > $CORPUSDIR/$DEVPREFIX.$L.numtoks
done

#Find most frequent number of tokens
paste $CORPUSDIR/$DEVPREFIX.$SL.numtoks $CORPUSDIR/$DEVPREFIX.$TL.numtoks | LC_ALL=C sort | LC_ALL=C uniq -c | sed 's:^[ ]*::' | tr ' ' '\t' | LC_ALL=C sort -k1,1 -n | tail -n 1 > $CORPUSDIR/$DEVPREFIX.mostfrequenttoknum

GREPEXPR=$(cut -f 2,3 $CORPUSDIR/$DEVPREFIX.mostfrequenttoknum)
GREPEXPR="	$GREPEXPR"

#Filter dev set
paste $CORPUSDIR/$DEVPREFIX.$SL $CORPUSDIR/$DEVPREFIX.$TL $CORPUSDIR/$DEVPREFIX.$SL.numtoks $CORPUSDIR/$DEVPREFIX.$TL.numtoks  | grep "$GREPEXPR"  | cut -f 1 > $CORPUSDIR/$DEVFILTERED.$SL
paste $CORPUSDIR/$DEVPREFIX.$SL $CORPUSDIR/$DEVPREFIX.$TL $CORPUSDIR/$DEVPREFIX.$SL.numtoks $CORPUSDIR/$DEVPREFIX.$TL.numtoks  | grep "$GREPEXPR"  | cut -f 2 > $CORPUSDIR/$DEVFILTERED.$TL
