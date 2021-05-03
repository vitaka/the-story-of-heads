#! /bin/bash

MYFULLPATH=$(readlink -f $0)
CURDIR=$(dirname $MYFULLPATH)

DEVPREFIX="dev.bpe"
DEVFILTERED="dev.bpe.samesize"

SL=$1
TL=$2
CORPUSDIR=$3

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
