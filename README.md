# kinyarwanda tweet sentiment

sorts kinyarwanda tweets into positive neutral or negative
compares simple word count models against a fine tuned transformer

| item | URL |
|---|---|
| live app | https://sentikinyarwanda.streamlit.app |
| model | https://huggingface.co/mbonikevin/afroxlmr-kinyarwanda-sentiment |
| demo video | link |

## why

kinyarwanda is spoken by about 12 million people but has no sentiment analysis
service without one reading public opinion at scale means reading by hand

## data

[afrisenti](https://github.com/afrisenti-semeval/afrisent-semeval-2023) kinyarwanda
part real tweets labelled by native speakers no ai generated data is used

| split | tweets | positive | neutral | negative |
|---|---|---|---|---|
| train | 3302 | 899 | 1257 | 1146 |
| validation | 827 | 225 | 315 | 287 |
| test | 1026 | 278 | 393 | 355 |

we keep the official splits so our scores can be compared with published results

two problems turned up in those splits see `src/leakage.py`

- 27 tweets sit in both train and test so a model can score on memorised text
- 7 of those carry a different label in each split which caps test accuracy at 99.3 percent

## models

baselines are most common class tfidf with naive bayes and tfidf with logistic
regression over word and letter patterns the last one is the real bar to beat

main model is [afroxlmr base](https://huggingface.co/Davlan/afro-xlmr-base) which
is xlm roberta trained further on 17 african languages

e5 is the key test afroxlmr and xlm r share a vocabulary and cut kinyarwanda into
the same 2.68 pieces per word so swapping in plain xlm r with nothing else changed
shows what the extra african pretraining is actually worth

## results

test set 1026 tweets macro f1 leads because the classes are uneven so accuracy can
look fine while a whole class is ignored

| model | macro f1 | weighted f1 | accuracy |
|---|---|---|---|
| most common class | 0.1846 | 0.2122 | 0.3830 |
| tfidf naive bayes | 0.5511 | 0.5532 | 0.5556 |
| tfidf logreg | 0.6017 | 0.5981 | 0.5984 |
| **afroxlmr base** | **0.6841** | **0.6802** | **0.6803** |

the 27 tweets shared between train and test barely matter scoring only the 999
unseen rows gives 0.6804 macro f1

e5 answers the question section 2 set up plain xlm r with an identical config
scores 0.5264 on validation against 0.6854 for afroxlmr the tokenizers are the
same so the 0.159 gap is the pretraining weights alone

## files

```text
src/data.py                 download clean split
src/leakage.py              shared tweets and clashing labels
src/tokenizer_analysis.py   how words get cut into pieces
src/baseline.py             the simple models
src/train_transformer.py    fine tuning and the experiment runner
src/error_analysis.py       what the model gets wrong and why
src/metrics.py              shared scoring
src/experiment_log.py       one row per run
src/viz.py                  charts
notebooks/                  runs everything needs a gpu for training
streamlit_app.py            the web page
reports/                    charts and result files
```

## running it

```bash
git clone https://github.com/mbonikevin/senti-kinyarwanda.git
cd senti-kinyarwanda
pip install -r requirements.txt
```

data tokenizer and baseline steps run on a normal laptop in about two minutes
training needs a gpu so open the notebook in colab and pick a t4 runtime the test
set is scored once at the end after every choice is made on validation
