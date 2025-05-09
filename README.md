# Chinese-Misogyny-Overview
Report: https://docs.google.com/document/d/15oAOzO4wyrgEr799Laxj4tNv7ho1KR7xVaFL6GQDXrE/edit?usp=sharing

## Data Acquisition
The SWSR dataset can be found here: https://github.com/aggiejiang/SWSR.git, published by Jiang et al. The other data are all acquired through Weibo API calls. First through acquiring weekly web-scraped hot topics, then acuiqring specific posts using the hot topics as query on the Weibo API call. 

## Data Cleaning
After removing hyperlinks, emojis, and hashtags, empty texts were dropped. 

## NLP models
Clustering, BERT, and LLMs were used on the same SWSR. A demo was made for Pinyin similarity feature engineering.
