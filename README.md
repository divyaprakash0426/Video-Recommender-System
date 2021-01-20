# Amazon Instant Video Recommender System
Recommendation systems can take advantage of social media in various ways. Recommendation systems are    defined as the techniques used to predict the rating one individual will give to an item or social     entity. These items can be books, movies, restaurants, and things on which individuals have different preferences. Prime Video, also referred to as Amazon Prime Video, is an Internet video on demand   service that is developed, owned, and operated by Amazon. In this project, various models are presented  such as Extra-TreesRegressor, XGBRegreesor, Matrix Factorization, SVD and Probabilistic Matrix   Factorization in predicting the rating and provide top video recommendations based on the predicted   ratings. Finally, the ensemble of top 2 model ratings is taken to get the top recommendations.   A recommendation system was built with better accuracy to predict the ratings, users would give for   videos which they haven′t seen before and to create a recommendation list for each user. A simple strategy to create such a recommendation list for a user is to predictthe ratings on all the items   that user didn′t buy before, then rank the items in descending order of the predicted rating, and   finally take the top items as the recommendation list.  

### Data  
Amazon Instant Video dataset available at: http://jmcauley.ucsd.edu/data/amazon/

### Algorithms Explored  
1. Matrix Factorization
2. SVD
3. Probabilistic Matrix Factorization
4. ExtraTrees Regressor
5. XGB Regressor

### Results  

| Model | Feature | MSE(train) | MSE(test) | Precision | Recall | F1 Score |
| ----- |-----    | -----      | --------- | ----------|------- | -------- |
| Matrix Factorization |UserId, VideoID | 0.0017 | 0.001 | 1.0 | 0.8547 | 0.921 |
| SVD |UserId, VideoID | 2.0247 | 2.060 | 0.961 | 0.396 | 0.560 |
| Probabilistic Matrix Factorization |UserId, VideoID | 0.0033 | 0.003 | 1.0 | 0.8448 | 0.9159 |
| ExtraTrees Regressor | UserId, VideoID, time, weekend_review | 0.006 | 1.306 | 0.845 | 0.7406 | 0.789 |
| XGB Regressor |UserId, VideoID, time, no. users, no. videos | 1.109 | 1.242 | 0.8355 | 0.8511 | 0.843 |
| Bagging(MF and PMF) | UserId, VideoID | 0.0019 | 0.0016 | 1.0 | 0.8612 | 0.925 |

# To run the program
1. Download data from [data](https://github.com/divyaprakash0426/Video-Recommender-System/tree/master/data)
2. Modify the following command in data_manipulation.py file [data_manipulation.py](https://github.com/divyaprakash0426/Video-Recommender-System/blob/master/code/data_manipulation.py)
```
def data():
    print('loading data...')
    initial_data = initial_data_from_dict('paste data file location here')

```
3. Run main.py file [main.py](https://github.com/divyaprakash0426/Video-Recommender-System/blob/master/code/main.py)