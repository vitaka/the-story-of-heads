#! /bin/bash
MYFULLPATH=$(readlink -f $0)
CURDIR=$(dirname $MYFULLPATH)

DEVPREFIX="devtranslated.bpe"
DEVFILTERED="devtranslated.bpe.samesize"

SL=$1
TL=$2
TRAINDIR=$3

CORPUSDIR="$TRAINDIR/corpus"

#Find translated dev set for best checkpoint
#TODO: extract update number from tune if the final BLEU is better
UPD=$( python $CURDIR/extract-update-num.py "$TRAINDIR/model/build/checkpoint/train.state-best_bleu.npz"  )

TRANSLATIONS="$TRAINDIR/model/build/translations/translations_$UPD.txt"

cut -f 1 $TRANSLATIONS > $CORPUSDIR/$DEVPREFIX.$SL
cut -f 2 $TRANSLATIONS > $CORPUSDIR/$DEVPREFIX.$TL

#Count tokens
for L in $SL $TL ; do
    cat $CORPUSDIR/$DEVPREFIX.$L | awk -F ' ' '{print NF}' > $CORPUSDIR/$DEVPREFIX.$L.numtoks
done

#Find most frequent number of tokens
paste $CORPUSDIR/$DEVPREFIX.$SL.numtoks $CORPUSDIR/$DEVPREFIX.$TL.numtoks | LC_ALL=C sort | LC_ALL=C uniq -c | sed 's:^[ ]*::' | tr ' ' '\t' | LC_ALL=C sort -k1,1 -n | tail -n 1 > $CORPUSDIR/$DEVPREFIX.mostfrequenttoknum

GREPEXPR=$(cut -f 2,3 $CORPUSDIR/$DEVPREFIX.mostfrequenttoknum)
GREPEXPR="\t$GREPEXPR"

#Filter dev set
paste $CORPUSDIR/$DEVPREFIX.$SL $CORPUSDIR/$DEVPREFIX.$TL $CORPUSDIR/$DEVPREFIX.$SL.numtoks $CORPUSDIR/$DEVPREFIX.$TL.numtoks  | grep "$GREPEXPR"  | cut -f 1 > $DEVFILTERED.$SL
paste $CORPUSDIR/$DEVPREFIX.$SL $CORPUSDIR/$DEVPREFIX.$TL $CORPUSDIR/$DEVPREFIX.$SL.numtoks $CORPUSDIR/$DEVPREFIX.$TL.numtoks  | grep "$GREPEXPR"  | cut -f 2 > $DEVFILTERED.$TL
