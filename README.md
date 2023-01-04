# Credit_Risk_Analysis_bootcamp

## Background and Purpose

"To loan or not to loan?", surely a question with which most lenders are familiar and wrestle frequently. With loans representing a large flow of wealth, it is wise to keep abreast of loan risk, and the increased digitization of loan records enables the automation of such risk assessment. To that end, machine learning is one way to tackle the automation. Below, we investigate currently-held loans, as of the first quarter of 2019, and test various machine learning algorithms for assessing the risk of a loan by the next payment period. We want to try singling-out the risky loans that present a potential loss of capital for our lendors.  

Thankfully for many lenders, the number of current loans vastly outnumbers the number of defaulted loans. This does present a problem for our machine learning purposes, however: when training the algorithm, there are very few datapoints to teach an algorithm what a bad loan looks like compared to good loans. Such datasets are known to be imbalanced; that is, there is an imbalance between the numbers of each outcome. There exist a number of ways to try re-balancing the training data to combat this issue, which we investigate here.  

We also must pay attention to how we grade our results. When we make predictions, we obviously either predict correctly or incorrectly. In the case of two outcomes (good loan or bad loan), this leads to four cases: two cases for correctly predicting either outcome and two cases for incorrectly predicting either outcome. Knowing we will make wrong predictions some of the time, we have two options: we care more about not incorrectly classifying good loans as bad loans or we care about identifying all bad loans even if a few good loans are mislabeled. Statiscally, these cases refer to the precision and sensitivy, respectively. Because a bad loan presents greater risk to our lendors, being a total loss of capital if it defaults, we focus on the latter option and grade our results with the sensitivity of our algorithms.

(It also pays to carefully note what we're studying here. Below, we use loans in which the issuees have begun making payments. This means that we're not predicting if a loan as a whole will be good or bad; we are actually asking if there is a risk the *next payment* on the loan will be late. However, even this interpretation is muddled because some loans here are late by a few days, a few months, or have already defaulted. We have classified all of these loans as high-risk compared to the current, or low-risk, loans. This would be important when trying to apply our models to non-historical data (i.e. making predictions for loan payments for which we do not already know the outcome). An already-late loan will have a late payment flag in this dataset but is trained herein as not-yet-late. This consideration should be taken into account for future projects.)


## Methods

### Data Preparation

*The data preparation steps below were performed identitically in each python notebook.*

#### Provided Cleaning Code

The csv dataset contains 144 columns and 115,675 entries. The provided code read in the dataset and performed numerous cleaning steps in the first cell.  
1. The second line reads in the data, skipping an unusable information link in the first row, and drops the last two columns (which are mainly filled with null values)
2. The third line selects only the columns of pre-determined interest defined above (the majority of these columns had null values for each row and are non-useful); 58 columns of data are dropped
3. The fourth line attempts to drop any columns in which *all* entries are null; no columns are removed from this step
4. The fifth line drops all rows that contain any null values; 33,138 rows are dropped
5. Lines six and seven masks (ignores) any loans that have the "Issued" status, or loans which have not had any payments paid yet; 13,720 rows are effectively removed
6. Lines eight and nine convert the interest rate column, initially represented as a string, to a numerical value
7. Lines ten through thirteen re-classify the loan status column (our target column) of rows to turn a 5-outcome problem into a binary (2-outcome) problem
    - Lines ten and eleven classify any loan that is current on repayment as a low-risk loan
    - Lines twelve and thirteen classify loans that are behind in payments (either in a grace period, late by 16-30 days, late by 31-120 days, or in default) as a high-risk loan
8. The final line simply resets the dataframe's index.

After the above steps, there are 86 remaining columns, each with 68,817 non-null values. However, additional cleaning steps can be taken.  

#### Single-valued Cases

Our machine learning algorithms will attempt to correlate different values between rows (in the same column) to the different outcomes. If a column is single-valued, there are no differences between rows to correlate against and the column serves simply as noise to our algorithms.  

To check for such columns, we use `pandas`' `duplicated()` function on the subset of each column (and sum the result). If there are 68,816 duplicates [to the first row] in a column, it means there is only a single value in the entire column.  

There are 10 single-valued columns that we drop.

#### Categorical Data

There are 76 columns remaining: 69 columns with numerical types and 7 columns with string types. Because our algorithms work best on numerical data, we must convert our string-typed columns to numerical data. To do this, we replace each (see below) categorical column with a binary matrix to represent each unique value. (Some columns, such as verification status and home ownserhip, could potentially be replaced with ordinal data. However, this requires a lot more discussion and knowledge of the dataset we do not have, so we forego such for our first foray into machine learning.)

For our 6 categorical feature columns, we use `pandas`' `get_dummies()` method to accomplish our goals. The last of the 7 categorical columns is actually our target column, which we can treat with ordinality, and we utilize `scikit-learn`'s `LabelEncoder` to transform this column. 

After these transformations, we have 86 numerical columns (float and integer representations) our algorithms will utilize. Importantly, for our loan status column, `high-risk` loans have been encoded as `0` and `low-risk` loans have been encoded as `1`. Therefore, when reading confusion matrices and classification reports, we focus on the `0` row.  

| outcome | encoding |
| ------- | :------: |
| **high-risk** | **0** |
| **low-risk** | **1** |

#### Splitting and Scaling

The loan data was split into training and testing data with `scikit-learn`'s `train_test_split`. The default split of 75% to training and 25% to testing was used; the `random_state=1` parameter was used for constistency; and no stratification was specified (therefore, none was performed). Of a total 68,817 observations, 68,470 loans were classified as low-risk and 347 as high-risk (0.504% are high-risk). Following the splitting:  
- Training data had 51,366 low-risk observations with 246 high-risk (0.477% of observations are high-risk)
- Testing data had 17,104 low-risk observations with 101 high-risk (0.587% of observations are high-risk)

Amongst the columns, there were a lot of different measurement ranges. For instance, the loan amount column had a range of 39,000 (st. dev. of over 10,000), but the interest rate column had a range of 0.25 (st. dev. of 0.05). To combat these range mismatches, the data was scaled with `scikit-learn`'s `StandardScaler`. This step transforms the data such that each column has a mean of zero and standard deviation of one.  

(An error of performing all of the logistic regression functions before scaling the data enables us to investigate the effect of scaling. The results can be found in `Unscaled_credit_risk_sampling.ipynb`. Indeed, we find that model results without scaling the data are less useful. The balanced accuracy scores for each model discussed below are at least 10% points worse for the unscaled data. Furthermore, the sensitivity of each model were also at least 8% points worse using the unscaled data.)

### Models and Adjustments

There were two model formats used, logistic regression and decision trees, as low-level comparisons among model types. For logistic regressions, the training data was adjusted for high-level comparisons among logistic regressions. No adjustments were made to the data for decision tree models. In each applicable step, the `random_state=1` parameter was set for consistency. 

#### Logistic Regression

Deliverables 1 and 2 of the challenge can be found in `credit_risk_resampling.ipynb` and involved performing logistic regressions (Logit functions) on the data to assess credit risk. The crux of the discussion is the imbalance between the number of low-risk and high-risk observations, ~0.5% of the total observations fall into the latter category, in the training data. Various methods were utilized to adjust this categorical split in the training data to compare the effects of sample population in predicting outcomes. For each of the below, `scikit-learn`'s `LogisticRegression` was fit to the adjusted training data and applied to the testing data.


<u>Non-Resampled</u>  
The first model is a non-resampled logistic regression (hereafter "NR Logit") to serve as a control. No over- or under-sampling was performed on this data, meaning the logistic regression was trained on 51,366 low-risk and 246 high-risk observations.   

<u>Random Over-Sampling</u>  
To compensate for the imbalance between the outcomes, one method is to randomly duplicate observations (an observation can be duplicated an indefinite number of times) in the minority outcome to balance the number of observations. In our naive Random Over-Sampling logistic regression (hereafter "ROS Logit") model, we used `imblearn`'s `RandomOverSampler` to do just that. Following resampling of the training data, there were 51,366 observations each of low- and high-risk outcomes. 

<u>SMOTE Over-Sampling</u>  
An algorithm called SMOTE,  Synthetic Minority Over-sampling TEchnique, was used to over-sample our minority outcome for a logistic regression model (hereafter "SMOTE Logit"). Rather than duplicating observations, as above, SMOTE creates sythentic (fake) observations based on real observations (via a nearest-neighbors heuristic). Using `imblearn`'s `SMOTE`, we again over-sample the high-risk observations so that there are 51,366 low- and high-risk observations. 

<u>Cluster Centroid Under-Sampling</u>  
The complement to over-sampling, under-sampling decreases the population of the majority outcome to match the population of the minority outcome, which we apply for our Cluster Centroid under-sampling logistic regression model (hereafter "CC Logit"). In this algorithm, the majority outcome observations are broken into groups, or clusters. Each observation in a cluster is then replaced in the training data by the center (e.g mean) of the cluster. Applying `imblearn`'s `ClusterCentroids` to our training data, the low-risk population has been reduced to that of the high-risk population, 246 observations. 

<u>SMOTEENN Under- and Over-Sampling</u>  
The last adjustment of the training data completed here is the idea of combining over-sampling the minority class and under-sampling the majority class, our SMOTEENN algorithm logistic regression (hereafter "SMOTEEN Logit"). The `SMOTEENN` algorithm from `imblearn` is a two-step process, applying the SMOTE algorithm above (over-sample) followed by an Edited Nearest Neighbors (ENN) algorithm (under-sample). The ENN works by comparing some observations to their nearest neighbor observations; if the outcomes of this subset of observations are different, this subset of observations is removed from the total set of observations. Applying the algorithm to our dataset may not have worked ideally, since the population of the majority class of low-risk observations was not decreased. This training set had 51,366 low-risk observations and 47,635 high-risk observations. 


#### Decision Trees  

Deliverable 3 of the challenge can be found in `credit_risk_ensemble.ipynb`, which focused on utiziling forests of decision trees to solve our classification problem. Random forest classifiers use many weak, shallow-depth decision trees -- an ensemble of trees -- to solve classification problems by effectively democratizing the classifiction to the weak trees. The two ensemble methods used here also attempt to address the imblanace in our data, though we don't need to adjust the training data like above. Each ensemble was fit with the training data and applied to the test data.

<u>Balanced Random Forest</u>  
Our Balanced Random Forest Classifier (hereafter "BRF Classifier") uses `imblearn`'s `BalancedRandomForestClassifier`. This classifier works by first taking a random sample of the minority observation class and then a sample in equal number of the majority class (much like the ROS Logit step, observations in the majority class can be sampled an indefinite number of times). The classifier then runs the decision tree algorithm on these samples before repeating from the first step.  

<u>Easy Ensemble AdaBoost</u>  
The Easy Ensemble AdaBoost Classifer (hereafter "EEAB Classifier"), by way of `imblearn`'s `EasyEnsembleClassifier`, works similarly to other random forest classifiers but with the addition of an AdaBoost step. First, the EEAB Classifier builds a classifier with the entirity of the minority population and a random sample of the majority population such that the number of observations in the majority sample is the total of the minority. A decision tree algorithm is run and labels are internally checked for correctness. In the AdaBoost step, if a training data observation was labeled incorrectly, the weight of that data point is increased for the next iteration of the algorithm, starting over at the first step.


## Results

With the models run on the testing data, we are now in a position to assess how well they performed by comparing to the known outcomes. The model's own score method is reported in the python notebooks but not reproduced here because they are intrinsically high (being an imbalanced dataset). The second assessment tool is `sklearn`'s `balanced_accuracy_score`, which reports a single score (0-1 scale): an average of the sensitivity of each class (i.e. high-risk and low-risk). Additionally from `sklearn`, we utilize `confusion_matrix` to produce the counts of correctly and incorrectly predicted outcomes. Lastly, `imblearn`'s `classification_report_imbalanced` presents a print-out of various metrics, including precision and sensitivity, among others (unused here), of each outcome.  

The table below presents an overview of each model's performance on the testing data. "BAS" is the balanced accuracy score, "TRH" is the number of correctly-predicted high-risk loans (out of 101 high-risk loans in the training set), "Prec" is the precision calculated on the high-risk outcome, and "Sens" is the sensitivity calculated on the high-risk outcome.

| Model | BAS | TRH | Prec | Sens | 
| ----- | :---: | :---: | ---: | :----: |
| NR Logit | 60.38% | 21 | 81% | 21% |
| ROS Logit | 83.27% | 83 | 3% | 82% |
| SMOTE Logit | 83.89% | 82 | 3% | 81% |
| CC Logit | 81.26% | 87 | 2% | 86% |
| SMOTEENN Logit | 83.89% | 83 | 3% | 82% |
| BRF Classifier | 78.12% | 69 | 3% | 68% |
| EEAB Classifer | 93.20% | 93 | 9% | 92% |  

### Features and Weights

The BRF Classifier enables us to get a feel for how the algorithms work by looking at the resulting weights of the features used. The `last_pymnt_amnt` (~6.8%), `total_rec_prncp` (~6.4%), `total_rec_int` (~6.2%), `total_pymnt` (~5.8%), and `total_pymnt_inv` (~5.5%) features had the greatest impacts, with `int_rate` (~3.9%) being the last feature above a 2.2% weight. Around 20 features had weights between 1% and 2% with almost all the others (over 55 features) falling below 1%, down to 0.01%. A single feature, `delinq_amt`, had a weight of 0.000% (unmeasurable to at least 8 decimal places).

Many of our formerly-categorical-turned-numerical (via `get_dummies()` above) features (e.g home ownership) had similar resultant weights: `home_ownership_OWN`, `home_ownership_MORTGAGE`, and `home_ownership_RENT` have weights of ~0.21%, ~0.21%, and ~0.20%, respectively. However, some formerly categorical features, such as the issue date column, saw a much more diverse split in weights: `issue_d_Jan-2019` had a weight of ~2.2%, `issue_d_Mar-2019` had a weight of ~1.3%, and `issue_d_Feb-2019` had a relatively measly weight of ~0.5%. This returns us to the previous conversation regarding ordinality of these categorical features: does equal weighting (e.g. home ownership) or unequal weighting (e.g. issue date) suggest a greater need for ordinality in converting these features? Alas, such a question is beyond the purview of this assignment and it shall remain unanswered.

## Discussion

The EEAB Classifier has the best scores across nearly all metrics and is the clear winner from our training and testing data (accordingly, it is the better of the two decision tree formats). Among just the logistic regression models, the CC Logit model was the most accurate at classifying the high-risk loans, but suffers a slight penalty on the BAS to the other adjusted-data models. With no surprise, the clear loser is the unuseful NR Logit model, which serves to highlight the importance of balancing the training data between the outcome classes.

Notably, the increased sensitivity of high-risk loans comes at the cost of decreased sensitivity of the low-risk loans (which is reflected in the decrease of precision of the high-risk outcome in the table above). Because the downside of a loan defaulting (i.e. total loss of capital) is greater than the downside of giving a low-risk loan a higher interest rate or loss of providing the loan, we take this as an acceptable consequence and strive for increased sensitivity of the high-risk loans. However, this also showcases the overall strength of the EEAB Classifier; not only did it have the highest sensitivity of the high-risk loans, it also had the highest sensitivity of the low-risk loans (again, reflected by the highest precision of the high-risk loans among the useful models).

Of course, there's no free lunch. The increased sensitivity of the high-risk loans correlates well with the amount of time and computing power required to run the model (including the training data adjustment). While no strict measurements were taken, the EEAB Classifer and CC Logit models took the longest to run. The BRF Classifier and SMOTEENN Logit models took a moderate amount of time and power, though these are not justified by the resulting sensitivity in the former's case. 

With the performance of the EEAB Classifier, the time and power necessary to run it is fully justified. After correctly classifying 93 of the 101 high-risk loans, missing only 8 of the 101 high-risk loans, and only incorrectly labeling 971 of 17,104 low-risk loans as high-risk, the EEAB Classifier is recommended to be used as a method to predict if currently-held loans will become high-risk.
