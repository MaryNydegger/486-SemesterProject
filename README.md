# Movie Review Sentiment Analysis

### Overview
* This project aims to perform sentiment analysis on a movie review dataset. The objective is to dive into sentiment (positive or negative) based on textual content using machine learning techniques.

### Goals
* *Problem Statement*: Predict sentiment polarity in moview reviews.
* *Dataset*: Utilizing a dataset containing 50,000 movie reviews - 25,000 for training and 25,000 for testing.
* *Methodology*: Using text preprocessing, feature engineering, and other machine learning models for analysis.

### Project Structure
* *Data*: The dataset consists of labeled moview reviews.
* *Preprocessing*: Cleaning, tokenization, and feature extraction from textual data.
* *Model Building*: Training and evaluating machone learning models for sentiment classification.
* *Evaluation*: Assessing model perforamnce using accuracy and other relevant metrics.
* The *workflow* follows this process: data loading and preprocessing, exploratory data analysis, model selection and training, evaluation and results, and then fine tune things for better performance.  


## Results 
* *Naive Bayes*: Validation Accuracy - 0.8524, Test Accuracy - 0.8406
* *Logistic Regression*: Validation Accuracy - 0.8764, Hyperparamter Tuning Validation Accuracy - 0.8764, Best Score - 0.8807
* *Random Forest*: Validation Accuracy - 0.8376, Hyperparameter Tuning Validation Accuracy - 0.8444, Best Score - 0.8807
* *AdaBoost Ensemble Method*: Accuracy - 0.8362
* *Voting Classifier Ensemble Method*: Accuracy - .7844

### Conclusion 
* *Insights*: Logistic Regression accuracy remained consistent between hyperparameter tuning and no tuning, whereas Random Forest accuracy slightly improved with the hyperparameter tuning. But the logistic regression model performed better and had a higher accuracy than random forest. Feature importance analysis reveladed key features contributing to model predictions. The AdaBoost method performed better than voting classifier in terms of accuracy. 
* *Areas for further improvement and refinement*: I want to collect additional data from webscraping to improve model performance and findings. I want to experiment with more advanced sentiment analysis techniques to try to yield better insights. I also want to explore more ensemble methods. 

##### Sourcing
* *Dataset source*: [https://ai.stanford.edu/~amaas/data/sentiment/]
* *Citing*: @InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}






