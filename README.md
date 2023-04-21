# Song Popularity Prediction

## Background

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


