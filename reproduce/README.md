# Reproduce
We try to reproduce the results in the paper titled "A Named Entity Based Approach to Model Recipes".

## Step 1
Create a prop.txt file in the train folder:
```
cd ./stanford-ner-2020-11-17/train
touch prop.txt
```
Add content to prop.txt
```
trainFile = train/ar_gk_train.tsv
serializeTo = trained_ar_gk.ser.gz
map = word=0,answer=1

useClassFeature=true
useWord=true
useNGrams=true
noMidNGrams=true
maxNGramLeng=6
usePrev=true
useNext=true
useSequences=true
usePrevSequences=true
maxLeft=1
useTypeSeqs=true
useTypeSeqs2=true
useTypeySequences=true
wordShape=chris2useLC
useDisjunctive=true
```
## Step 2
Train it, using:
```
java -cp "stanford-ner.jar:lib/*" -mx4g edu.stanford.nlp.ie.crf.CRFClassifier -prop train/prop.txt
```

## Step 3
Evaluate it, using our ipynb file `./stanford-ner-2020-11-17/ReproduceStanfordNER.ipynb`.

## Results
We train the model with ar_gk_train.tsv. The results reported is on ar_gk_test.tsv.
|         | Recall | Precision | F1 Score |
|---------|--------|-----------|----------|
|DF       | 0.4235 | 0.4301    | 0.4267   |
|NAME     | 0.6454 | 0.6448    | 0.6451   |
|O        | 0.6007 | 0.5979    | 0.5993   |
|QUANTITY | 0.6437 | 0.6423    | 0.6430   |
|SIZE     | 0.3269 | 0.3333    | 0.3301   |
|STATE    | 0.4876 | 0.4850    | 0.4863   |
|TEMP     | 0.3488 | 0.3333    | 0.3409   |
|UNIT     | 0.6269 | 0.6369    | 0.6319   |
|Overall Test Accuray | 0.6083            |


# Reference:
We follow the instructions here [1].

- [1]: Stadford NER CRF: https://nlp.stanford.edu/software/crf-faq.html#a