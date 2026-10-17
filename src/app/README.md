---
title: Kinyarwanda Tweet Sentiment
emoji: 🇷🇼
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
---

# kinyarwanda tweet sentiment

afroxlmr base fine tuned on the afrisenti kinyarwanda subset
sorts a tweet into positive neutral or negative

code and report https://github.com/mbonikevin/senti-kinyarwanda

to deploy this space you need `app.py` `requirements.txt` this readme and a copy
of `src/data.py` which holds the same cleaning used in training

```bash
mkdir -p src && cp ../src/data.py src/ && touch src/__init__.py
```
