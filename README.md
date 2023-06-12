# Song Popularity Prediction

## Introduction

For musicians, record companies, and music streaming services, a song's popularity is essential since it can decide the success of a new release or even an entire career. Professionals in the music business can plan their marketing, promotion, and distribution strategies by forecasting which songs are most likely to catch on. Researchers and music industry professionals alike have a unique opportunity to investigate and comprehend the aspects that influence song popularity and trends by analyzing huge music datasets. By utilizing these resources, stakeholders can develop new winning strategies for an extremely competitive market and make data-driven decisions.

The goal of this project is to create a machine learning model that can precisely predict a song's popularity based on its acoustic characteristics and artist’s information. Metrics including accuracy, precision, recall, and F1 score will be used to assess the model's performance.

### Significance

The music industry professionals can save time and money if song popularity can be predicted accurately, making this issue crucial. They can utilize machine learning models to forecast the success of a new song or album before it is released instead of depending on trial-and-error techniques. This can lessen financial risk and raise the likelihood of success in the cutthroat music business.

### Project Potential
Some instances where our project might be crucial is:

- The music industry professionals can use their resources more efficiently if they can accurately predict a song's level of popularity. The music industry may concentrate on advertising songs that are expected to be popular rather than spending money promoting
a song that may or may not be popular.

- The music industry can enhance its artist choosing process with precise song popularity
estimates. This can assist the industry in identifying fresh potential and helping it decide
whether to support a given artist.

- The music industry can boost the amount of streams and downloads, which will increase
income, by marketing songs that are expected to be popular. Resulting in a boost of income.

### Data Sources
We are utilizing the dataset produced by the Brazilian Federal University of Minas Gerais' computer science department. The dataset comes in a zip format divided into 3 directories — metadata, popularity, and song features. These directories contain multiple CSVs representing different tables of the MusiOSet dataset. The dataset contains more than 20,000 rows that are dispersed across several tables and we merge all the tables to obtain our initial data collection.

Link: https://marianaossilva.github.io/DSW2019/#downloads

### Execution

We divided the project into 3 phases : 
1. Exploratory Data Analysis
2. Applying Machine Learning Algorithms and Statistical Models
3. Build Data Projects for Presentation of Findings

## Phase 1 - Exploratory Data Analysis

### Data Cleaning

1. Dropping null values : Here in this project we are dropping the null values or NA values from the songs_pop, songs_df and artists_pop dataframes.

2. Checking for the types of rows for cleaning

3. Merging tables : We mainly merge the songs info and popularity dataframes. Then till is merged with tracks and acoustic features dataset. After this we merge the resulting dataset with artist info and artist popularity. 

4. Drop duplicates : Duplicate rows from the songs popularity dataframe are removed.

5. Dropping irrelevant columns : The below columns are removed from the final dataset as they are not useful in the prediction process.

6. Type change from object to datetime : The release_date variable is an object initially when it is read but for further tasks, it is converted to datetime format.

7. Finding unique values : Only the unique entries of songs in popularity table are useful. Hence we first check the count of the unique songs present in the dataset.

8. aggregate (groupby) to find popularity of artist : Each artist has a popularity score for each year he is present in the dataset along with if he/she is popular or not. We only need to know if the artist is popular or not. Hence to get a single value of popularity we take the majority count of popularity of the artist to determine this.

9. df.apply() to convert object of dict to string to get artist id for merging : To merge songs and artists dataframe, we need the artist_id in the songs dataframe. But this is present in the dict structure of type {“artist_id”: “artist_name”}. Hence we create a new column by extracting the key of this structure. This structure then is added back to the songs dataframe.

10. Removing outliers : There are very few songs which can go on for more than 13 mins. Hence to remove these outliers, we filter the songs dataframe to only keep songs less than 13 mins.

### EDA

1. Checking Main statistics of the data using df.describe() : 

![alt text](https://github.com/adiimated/Song-Popularity-Prediction/blob/main/dataset/images/dfdescribe.png)

To get a sense of spread of the data, central tendency, dispersion, and shape of the distribution of the data in a DataFrame, we use df.describe().

2. Plotting a heatmap of Correlation Matrix : To get the correlation between the different variables in the dataset, we plot the heatmap for the dataset.

Some of the strongly correlated features are: 

➔ (speechiness, explicit) : Explicit lyrics frequently contain more spoken words which are more likely to include profanity or other types of explicit language, which may account for the association between speechiness and explicitness in music. Also, as they concentrate on telling stories and delivering a message, songs with higher speechiness scores may be more likely to have explicit lyrics.

➔ (energy, loudness)
Loud noises can evoke feelings of enthusiasm and intensity, which are frequently connected with energetic music. Similar to this, energetic music typically has a quick tempo and a powerful beat, which frequently results in louder volumes.

➔ (danceability, valence)
The fact that happy and uplifting music tends to be more danceable can be used to explain the relationship between danceability and valence. Songs with a high valence, which is a measure of how good or negative the emotions the music evokes, are frequently characterized by a quick tempo and a powerful beat, making them better suited for dancing. Similar to this, danceable music has an optimistic, uplifting atmosphere that might heighten the song's valence.

3. Bar Graph -- Popularity of Songs : We created a bar graph showing how many popular and unpopular songs are included in the collection. In our dataset, there are more songs that are not popular than popular songs.

4. Line graph -- Trend of songs, non-explicit and explicit over the years : The graph demonstrates a pattern where the number of songs without explicit language or topics declines with time; it suggests that there has been a fall in the creation or release of such songs. Again, there has been a noticeable rise in the number of songs containing explicit material over time. The number of explicit songs climbed dramatically with each passing year. This might be the result of a number of things, including shifting societal norms and attitudes toward explicit language and themes in music, technological developments, and easier distribution and accessibility of music.

5. Bar graph -- Top 10 genres of all time

![alt text](https://github.com/adiimated/Song-Popularity-Prediction/blob/main/images/eda/eda_top10.jpg)

Depending on variables including geography, demographics, and cultural trends, the popularity of certain genres can vary dramatically. Considering these factors, the most listened to genres are:
* Album rock
* Adult standards
* Dance Pop
* Contemporary Country
* Classic Soul
* Brill Building Pop
* Disco
* Karaoke
* Bubblegum Pop
* Alternative Metal


6. Line Graph with Hue -- Song Year End Score VS Year Based on Popularity of Songs : 

![alt text](https://github.com/adiimated/Song-Popularity-Prediction/blob/main/images/eda/eda_yearendscore.jpg)

Songs published more recently may have been more well-liked or streamed more frequently than songs released in years past, considering that the year-end score for songs has been rising over time. A number of things, such as modifications in consumer behavior, technological developments, and alterations in music business trends, could be to blame for this. It's crucial to remember that year-end rankings can not always be a true representation of the overall excellence or worth of a song, as popularity can be influenced by a number of variables, including marketing, social media presence, and the date of release. Generally, a rising year-end score implies that more recent songs have become more well-known or popular than songs from earlier years.

7. Histogram -- Duration of all songs : If the distribution of song lengths on the graph is any indication, the most typical song length is probably between 3 and 4 minutes, with the majority of songs falling between 187000 and 260000 milliseconds. This song's typical length can be attributed to a variety of factors. For instance, a song lasting three to four minutes can convey a full musical concept, tell a story, or express an emotion while being brief enough to keep listeners interested and fit into the conventional structure of a commercial radio broadcast. Overall, the concentration of songs in the 3–4 minute range indicates that this is a well-liked and effective formula for writing music that connects with listeners and satisfies industry standards.

8. Scatter Plot -- Loudness VS Energy Based on Popularity of Songs

![alt text](https://github.com/adiimated/Song-Popularity-Prediction/blob/main/dataset/images/LOUDNESS%20VS%20ENERGY%20BASED%20ON%20POPULARITY%20OF%20SONGS.png)

We can see from the graph above that even if the music is loud and energetic, we cannot predict whether or not it will be popular. But we can definitely say that loudness and energy are 2 factors that are highly correlated. This could be due to the fact that loudness can evoke feelings of excitement and intensity, whereas energy is frequently linked to a song's tempo and rhythm. When these elements work together, they can have a strong emotional effect on listeners and increase a song's memorability and appeal.
 
 9. Box Plot -- loudness, acousticness, danceability, valence

![alt text](https://github.com/adiimated/Song-Popularity-Prediction/blob/main/images/eda/eda_boxplot.jpg)

Valence:
We can see from empirical data that the IQR of popular songs is typically narrower than the IQR of all songs combined. As a result, there are less extreme valence scores and a greater consistency in the emotional content of popular songs. In comparison, all songs often have a wider range of valence scores and a more diversified emotional content.
  
Popular songs are often written to appeal to a wide audience, which may necessitate a more consistent emotional tone, which may be one explanation for this discrepancy. In contrast, a broader variety of genres, styles, and moods are present in all songs as a whole, which inevitably leads to a wider variety of emotional content.


Danceability:
There may be a number of reasons why the IQR of popular songs differs from all other songs. Dancing is frequently connected to fun and amusement, therefore it stands to reason that successful songs are more likely to be produced and marketed with danceability in mind. Another influence might be the prevalence of popular music in social settings where dancing is widespread, including parties and clubs. Songs that are easier to dance to might therefore have a higher chance of being well-known and generally recognized.

Acousticness:
The interquartile range of popular songs differs significantly from the interquartile range of all songs, it may be that popular songs and non-popular songs have different acoustic characteristics.
It might be the case, for instance, that popular songs have more recurrent patterns in their acoustic properties or a narrower spectrum of acoustic features than non-popular songs. However, it's possible that songs that are more popular have a greater variety of acoustic characteristics than songs that are less popular, but that there are some essential characteristics that consistently set them apart from other songs.

Loudness:
The IQR of popular songs is lower than the IQR of all songs combined, it could mean that the loudness levels of popular songs are more consistently high.
This might be as a result of the more uniform loudness level across popular songs, which are made to be more approachable and catchy.
   
10. Radar graph -- showing mean of features for top 100 songs and the rest of the dataset : The acousticness, danceability, energy, instrumentality, liveliness, speechiness, and valence mean for the top 100 songs in popularity as well as the remainder of the dataset are all displayed in the radar graph above. We may deduce from this graph that the top 100 songs have higher energy and danceability than the remainder of the dataset. We also see that the top 100 songs have significantly less liveness, valence, and acoustic content than the remainder of the sample.

![alt text](https://github.com/adiimated/Song-Popularity-Prediction/blob/main/dataset/images/Radar%20graph.png)

## Phase 2 - Applying Machine Learning Algorithms and Statistical Models

Models Used :
The below models were trained on the training data for the purpose of predicting the popularity of the test data songs.

● Logistic Regression

● Neural Network

● Neural Network with L2 regularization

● Random Forest

● SVM

● KMeans

### Logistic Regression

Introduction:

Logistic regression is a type of statistical model used for binary classification, which means predicting one of two possible outcomes. It uses a logistic function, also known as a sigmoid function, to model the probability of the positive class as a function of the input features. The logistic function maps any input value to a probability value between 0 and 1.1.

Why Logistic Regression ?

1. Our goal is to divide music into two categories, popular and unpopular, which calls for binary classification. Logistic regression is a good option for this assignment because it is made expressly for binary classification issues.

2. Also, Logistic regression provides interpretable results, allowing you to understand the relationship between input features and the output prediction. This can be useful when trying to analyze how different song attributes (e.g., tempo, energy, danceability) affect user preferences.

3. Logistic regression can serve as a good baseline model. We can compare the performance of logistic regression with more complex models (NN models) to decide if the increased complexity is worth the trade-off.

Training / Tuning the Model :

To tune/train the model we did the following work (not including the work in phase 1) :
1. Load Data: The data was loaded from the songs_df dataframe, and relevant columns were
selected based on their categories.

2. Data Preprocessing:
a. rank_score column was converted into a binary classification problem by setting a threshold of 66.5, and values above the threshold were labeled as 1 and below as 0.
b. Categorical columns were encoded using label encoding or one-hot encoding (based on user preference).
c. Numeric columns were standardized using StandardScaler.

3. Train-Test Split: The preprocessed data was split into training and testing sets, with 30%
of the data used for testing and a random state of 42.

4. Model Building: A logistic regression model was built using LogisticRegression from
sklearn.linear_model. The maximum number of iterations was set to 10,000.

5. Model Evaluation: The model's accuracy was evaluated using accuracy_score from sklearn.metrics. Additionally, the classification_report was generated to evaluate the precision, recall, and f1-score for both classes, and confusion_matrix was generated to show the number of true positives, false positives, true negatives, and false negatives.

7. Finally, the seaborn and matplotlib.pyplot libraries were used to generate a confusion
matrix heatmap to visualize the results.

Effectiveness of the model:

The model achieved an overall accuracy of 0.78, which means it correctly classified 78.3% of the cases. The precision for class 0 is 0.79, which means that out of all the cases that the model classified as class 0, 79% were actually class 0. The recall for class 0 is 0.98, which means that out of all the cases that are actually class 0, the model correctly identified 98% of them.

The precision for class 1 is 0.53, which means that out of all the cases that the model classified as class 1, 53% were actually class 1. The recall for class 1 is 0.08, which means that out of all the cases that are actually class 1, the model correctly identified only 8% of them. The F1-score is a measure of the model's overall performance, taking into account both precision and recall. The F1-score for class 0 is 0.88 and for class 1 is 0.14.

The support column shows the number of instances in each class. There are 4,781 instances of class 0 and 1,335 instances of class 1. The macro average of precision, recall, and F1-score is calculated as the average of the scores for both classes, without considering their distribution. The macro average precision, recall, and F1-score in this case are 0.66, 0.53, and 0.51 respectively.

The weighted average of precision, recall, and F1-score is calculated as the weighted average of the scores for both classes, taking into account their distribution. The weighted average precision, recall, and F1-score in this case are 0.73, 0.78, and 0.71 respectively.
