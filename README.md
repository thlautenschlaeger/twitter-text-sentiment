# twitter-text-sentiment
Bot that learns text sentiment analysis to positive and negative tweets with implied neutral ratings. 
The bot gets trained with a training set and evaluates on live tweets.

## Example usage
```
python examples/train_and_analyze.py
```

This script trains a random forest algorithm and performs 
sentiment analysis on the given keyword. Add your twitter api key as **key.json** 
to the root folder. Feel free to play around with this script and tweak 
some hyperparameters etc.

## Samples
The samples are from: 
[https://github.com/nunomroliveira/stock_market_sentiment](https://github.com/nunomroliveira/stock_market_sentiment)

