# Data Science Coding Test

### Description
This project focus on a dataset about vedio comments with the following three columns:
1. creator_name. Name of the YouTube channel creator.
2. userid. Integer identifier for the users commenting on the YouTube channels.
3. comment. Text of the comments made by the users.

5,819,470 reviews are included.


### Step 1: Identify Cat And Dog Owners: Identify users who are definitely cat and/or dog owners, and which kind of pet they own.

For this step, we labeled the first 100,000 reviews (~1/60 of the dataset) and labeled the cat/dog owners by a coarse classifier. The reviews were tokenized as sentences and further as words with the package of NLTK. The classifier is defined as a filter: if the some key words, like 'dog'/'dogs', 'my'/'own', 'I'+'have', 'we'+'have' appears in one sentence, this review is labeled as a dog owner. The key words were selected basing on the dataset. 865 reviews were labeled as 'dog owner', and 453 reviews were labeled as 'cat owner'. The false positive rate is around 10% according to manual checking, and the false positive labels were corrected.


Codes: labelTrainingSet.ipynb

Input: input/animals_comments.csv

Output: output/labeledData.csv

output/labeledData_cleaned.csv (after manual checking)

### Step 2: Build And Evaluate Classifiers: Build classifiers for the labeled cat / dog owners and measure the performance of the classifiers.

### Step 3: Classify All The Users: Apply the cat / dog classifiers to all the users in the dataset (including unlabeled). Estimate the fraction of all users who are actually cat/dog owners taking into account false positives.

In this step, the text in reviews are first represented with 'one-hot representation' method, where dictionary was built with scikit-learn package. The text dataset was converted to a large sparse matrix. Therefore, XGBoost classifier was used as it takes special care for sparsity-aware split finding. XGBoost model was trained with the labeled dataset from Step 1, and five fold cross validation was used. This model achieved the accuracy score of 0.993 in test set.

Among all users, 0.9744% are labeled as cat owners by the model, while 1.8833% are labeled as dog owners. Additionally, the false positive rate is around 10% in Step 1, even though most of it was corrected manually. Therefore, the fraction of all users who are actually cat owners is at least 0.009744 * 0.9 * 0.993 = 0.87 %, and the fraction of all users who are actually cat owners is at least 0.018833 * 0.9 * 0.993 = 1.68%.

Codes: labelTrainingSet.ipynb

Input: output/labeledData_cleaned.csv

Output: output/animals_comments_labeled.csv

### Step 4: Extract Insights About Cat And Dog Owners: Find the top 5 (interpretable) features, which differentiate cat and dog owners.

In order to find the top features, which differentiate cat and dog owners, we train another XGBoost classifier for the owner data to distinguish dog owners and cat owners, and find that:
1. In comments, cat/dog owners mentioned different key words, like 'kitty', 'persian', 'fish', 'puppy', 'kittens'.
2. In average, dog owners have 1.163 comments per ID, while cat owners have 1.152 comments per ID. The users owning both dog and cat have 3.68 commnents per ID. Dog owners comment more. 
3. In average, dog owners leave comments for 1.062 creators, while cat owners leave comments for 1.058 creators. Dog owners pay attention to more creators. 

Codes: labelTrainingSet.ipynb

### Step 5: Identify Creators With Large Cat And Dog Ownership Audience: Find the top 10 creators with the largest audience fractions for cat or dog owners. Estimate the errors on the audience fractions and do not include creators with unreliable measurements.

The creators with fewer than 50 user IDs were filtered. Error is smaller than 1 - 0.9 * 0.993 = 10.6%. 
The top 10 creators are TV BINI, Calm My Dog House, Brent Atwater, Pawfessor, Larry Krohn, Rachel Fusaro, Zak Georges Dog Training rEvolution, Dog Training by Kikopup, Floppycats.com, V4VLuLz GAMES.

Codes: labelTrainingSet.ipynb

### Discussion
1. The error in the first step could be reduced by manual correction. It is the main error in the above analysis, but it could be easily reduced. 
2. Other classifiers could also be tried in this problem, e.g. SVM classifier, logistic classifier or neural network. Undersampling may be done before training these models for unbalanced data.
 
 
 Note: the input and some output dataset are too large to be uploaded.
