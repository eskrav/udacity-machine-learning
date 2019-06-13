
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime as dt
from scipy import stats as st

import matplotlib.pyplot as plt
import seaborn as sns
import visuals as vs

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

pd.options.display.max_columns = None

import warnings
warnings.simplefilter('ignore')
import dill


# ## Exploring the Data

# In[2]:


data = pd.read_csv("data/prosperLoanData_clean.csv")


# The data I've imported above is pre-cleaned, and has been previously explored and summarized.  See https://eskrav.github.io/udacity-data-analyst/explore-and-summarize/explore-and-summarize.html, for a full summary of the cleaning process and a visual exploration of the dataset.  
# 
# All currently active loans have already been removed from the dataset, as I am interested only in historical loans whose ultimate fate (repayment or default) is already known.  The few loans which were cancelled have been removed from the data, as they do not result in any gain or yield for the lender, and therefore are not of as much interest to the problem at hand, which is helping lenders deciding where to lend their money.
# 
# Several new features have already been added to the original data - for example, simple string and numerical binary factors indicating whether the loan was completed or not; a continuous feature indicating what percent of their investment the lenders in fact earned back on the historical loans; and so forth.  Many of these features I will remove early on in this process, as they are redundant with other features, or of no relevance to the task at hand (e.g., they concern information that principally cannot be available to the lender at the time that they are making their decision).
# 
# In total, the dataset contains 55084 data points corresponding to individual loans, with 32 continuous and 23 categorical features.  Although the data has already been cleaned, there is still missing data that was not imputed, and non-normally-distributed continuous features which will need to be transformed.  Continuous features will also need to be scaled.

# In[3]:


data.sample(10)


# Based on my previous exploration of this data, I am immediately removing those features which are irrelevant to the task at hand, or redundant with other features:
# 
# The following are internal loan and borrower identifiers, which are largely redundant with other identifiers, and are of no independent use: `ListingKey`, `ListingNumber`, `LoanKey`, `LoanNumber`, `MemberKey`. Of these, I will keep only `ListingKey` and `MemberKey` for the purposes of identifying individual loans and borrowers, until I have cleaned the data.
# 
# The following concern internal group membership, about which there is no further information available (furthermore, most loans do not belong to groups); I will therefore leave these aside for the time being: `CurrentlyInGroup`, `GroupKey`.
# 
# The following concern information that is not yet available to the lender at the time they are making the decision of whether to lend money, and therefore cannot serve as a useful feature for the lender, or is not directly informative about whether the loan is closed, or not: `ClosedDate`, `LP_CustomerPayments`, `LP_CustomerPrincipalPayments`, `LP_InterestandFees`, `LP_ServiceFees`, `LP_CollectionFees`, `LP_GrossPrincipalLoss`, `LP_NetPrincipalLoss`, `LP_NonPrincipalRecoverypayments`, `LoanCurrentDaysDelinquent`, `LoanFirstDefaultedCycleNumber`, `LoanMonthsSinceOrigination`, `PercentYield`.
# 
# The following are redundant with the `Rating` feature, which combines pre- and post-2009 Prosper rating schemes, as described in the previous data summary: `ProsperRating.num`, `ProsperRating.alpha`, `CreditGrade`.
# 
# The following are redundant with the label of interest (`Completed.num`), which indicates whether the loan was repaid in full, or defaulted: `LoanStatus`, `Completed`.  The following are redundant with `CreditScore`, which imputes a single average expected credit score (with minor loss of information): `CreditScoreRangeLower`, `CreditScoreRangeUpper`.  The following is redundant with the loan origination date: `LoanOriginationQuarter`.  The following is redundant with the borrower rate: `BorrowerAPR`.  The following is redundant with monthly income, which as a continuous feature is more easily interpretable: `IncomeRange`.  The following is redundant with age of credit history at the time the loan was originated: `FirstRecordedCreditLine`.
# 
# Almost all loans are completely funded (`PercentFunded`).

# In[4]:


redundant_irrelevant = ["ListingNumber", "LoanKey", "LoanNumber", "LoanStatus", "ClosedDate",
                       "ProsperRating.num", "ProsperRating.alpha", "CreditScoreRangeLower", "CreditScoreRangeUpper",
                       "LoanOriginationQuarter", "LP_CustomerPayments", "LP_CustomerPrincipalPayments",
                       "LP_InterestandFees", "LP_ServiceFees", "LP_CollectionFees", "LP_GrossPrincipalLoss",
                       "LP_NetPrincipalLoss", "LP_NonPrincipalRecoverypayments", "PercentFunded", "PercentYield",
                       "Completed", "LoanCurrentDaysDelinquent", "LoanFirstDefaultedCycleNumber", 
                       "LoanMonthsSinceOrigination", "BorrowerAPR", "CurrentlyInGroup", "GroupKey", "IncomeRange",
                       "CreditGrade", "FirstRecordedCreditLine"]

data.drop(redundant_irrelevant, axis=1, inplace=True)


# In[5]:


data.info()


# First, I convert the dates to continuous timestamps, as the methods I use are not able to handle datetime objects:

# In[6]:


data['ListingCreationDate'] = pd.to_datetime(data['ListingCreationDate']).astype(np.int64)
data['DateCreditPulled'] = pd.to_datetime(data['DateCreditPulled']).astype(np.int64)
data['LoanOriginationDate'] = pd.to_datetime(data['LoanOriginationDate']).astype(np.int64)


# Below, I examine features with null values, and infer from existing values if they can/should be imputed (e.g., an `NA` may represent a 0 for some features, if 0's are not generally coded, but would be expected to show up naturally):

# In[7]:


data_num_missing = data.isna().sum()
data_num_missing


# In[8]:


# percent of data missing per label:
data.isna().mean().round(4) * 100


# In[9]:


# value counts for feature of choice
data.TradesOpenedLast6Months.value_counts()


# In[10]:


data.describe()


# ### Data Imputation

# `EstimatedEffectiveYield` was a measure not available prior to mid-2009.  However, it measures the borrower rate minus expected fees, which can be imputed.  Given the data is not normally distributed (which will be examined further below), I will impute missing data with the median difference between borrower rate and estimated effective yield, as implemented in https://www.kaggle.com/jschnessl/prosper-analysis/notebook:

# In[11]:


data[data.EstimatedEffectiveYield.isna()].sample(5)


# In[12]:


estimated_fees = (data["BorrowerRate"] - data["EstimatedEffectiveYield"]).median()
data["EstimatedEffectiveYield"].fillna(data["BorrowerRate"] - estimated_fees, inplace=True)
data["EstimatedEffectiveYield"].isnull().sum()


# It's unclear how `EstimatedLoss` is computed.  Given the potential importance of this feature, and the difficulty in imputing it -- as well as the clear dependance on other features, such as risk measures -- I will not attempt to impute this feature, and will look for an algorithm robust to missing values, or simply leave this feature out.  The other potential solution would be to analyze pre-mid-2009 and post-mid-2009 datasets separately.  Ideally, one may want to do linear regression to infer likely values, but that is currently beyond the scope of this project.

# In[13]:


data[data.EstimatedLoss.isna()].sample(5)


# `EstimatedReturn` is the difference between estimated yield and estimated loss, and cannot be computed without the latter.  I will treat this feature similarly to `EstimatedLoss`.

# In[14]:


data[data.EstimatedReturn.isna()].sample(5)


# `ProsperScore` is a post-2009 measure.  In my exploratory analysis, it seemed that it might not be entirely redundant with the various Prosper ratings.  I don't believe there is any good way to impute this feature, and will either omit it from the model, or use a model robust to missing values.

# In[15]:


data[data.ProsperScore.isna()].sample(5)


# `BorrowerState` cannot be imputed.  Unknown values can be labeled as such; however, given the number of feature levels, and no clear expectations for how this feature will affect reliability, the states may need to be grouped into more meaningful geographical regions or categories prior to use.  I leave this for future work, and remove this label for the time being.

# In[16]:


data[data.BorrowerState.isna()].sample(5)


# In[17]:


data.drop(columns="BorrowerState", axis=1, inplace=True)


# `Occupation` cannot be imputed.  Unknown values can be labeled as such, but as with state, known values would need to be grouped meaningfully (e.g. by working class, minimally) to be informative.  I likewise leave this for future work, and for now drop this feature from the dataset.

# In[18]:


data[data.Occupation.isna()].sample(5)


# In[19]:


data.drop(columns="Occupation", axis=1, inplace=True)


# `EmploymentStatus` cannot be imputed.  Unknown values will be labeled as such.

# In[20]:


data[data.EmploymentStatus.isna()].sample(5)


# In[21]:


data["EmploymentStatus"].fillna("Unknown", inplace=True)
data["EmploymentStatus"].isnull().sum()


# I will impute `EmploymentStatusDuration` with the median value for this variable.  I would also consider dropping all missing rows, potentially, expecially if there were also missing values in other difficult-to-impute features.  Another possibility, to be explored in the future, is using linear regression to impute these values based on features judged likely to be relevant (e.g., income, occupation, home ownership).  Alternately, an algorithm robust to missing values may be able to handle this data.

# In[22]:


data[data.EmploymentStatusDuration.isna()].sample(5)


# In[23]:


median_employment_status_duration = data["EmploymentStatusDuration"].median()
data["EmploymentStatusDuration"].fillna(median_employment_status_duration, inplace=True)
data["EmploymentStatusDuration"].isnull().sum()


# It's unclear what the right course of action here is, but I will impute missing `CurrentCreditLines` with median values.

# In[24]:


data[data.CurrentCreditLines.isna()].sample(5)


# In[25]:


median_credit_lines = data["CurrentCreditLines"].median()
data["CurrentCreditLines"].fillna(median_credit_lines, inplace=True)
data["CurrentCreditLines"].isnull().sum()


# Same for `OpenCreditLines`.

# In[26]:


data[data.OpenCreditLines.isna()].sample(5)


# In[27]:


median_open_credit_lines = data["OpenCreditLines"].median()
data["OpenCreditLines"].fillna(median_open_credit_lines, inplace=True)
data["OpenCreditLines"].isnull().sum()


# Same for `TotalCreditLinespast7years`.

# In[28]:


data[data.TotalCreditLinespast7years.isna()].sample(5)


# In[29]:


median_7_credit_lines = data["TotalCreditLinespast7years"].median()
data["TotalCreditLinespast7years"].fillna(median_7_credit_lines, inplace=True)
data["TotalCreditLinespast7years"].isnull().sum()


# I will impute missing values in `InquiriesLast6Months` with median values.

# In[30]:


data[data.InquiriesLast6Months.isna()].sample(5)


# In[31]:


median_inquiries = data["InquiriesLast6Months"].median()
data["InquiriesLast6Months"].fillna(median_inquiries, inplace=True)
data["InquiriesLast6Months"].isnull().sum()


# I will impute missing values in `TotalInquiries` with median values.

# In[32]:


data[data.TotalInquiries.isna()].sample(5)


# In[33]:


median_total_inquiries = data["TotalInquiries"].median()
data["TotalInquiries"].fillna(median_total_inquiries, inplace=True)
data["TotalInquiries"].isnull().sum()


# I will impute missing values in `CurrentDelinquencies` with median values.

# In[34]:


data[data.CurrentDelinquencies.isna()].sample(5)


# In[35]:


median_delinquencies = data["CurrentDelinquencies"].median()
data["CurrentDelinquencies"].fillna(median_delinquencies, inplace=True)
data["CurrentDelinquencies"].isnull().sum()


# I will impute missing values in `AmountDelinquent` with median values.

# In[36]:


data[data.AmountDelinquent.isna()].sample(5)


# In[37]:


median_amount_delinquent = data["AmountDelinquent"].median()
data["AmountDelinquent"].fillna(median_amount_delinquent, inplace=True)
data["AmountDelinquent"].isnull().sum()


# I will impute missing values in `DelinquenciesLast7Years` with median values.

# In[38]:


data[data.DelinquenciesLast7Years.isna()].sample(5)


# In[39]:


median_7_delinquencies = data["DelinquenciesLast7Years"].median()
data["DelinquenciesLast7Years"].fillna(median_7_delinquencies, inplace=True)
data["DelinquenciesLast7Years"].isnull().sum()


# I will impute missing values in `PublicRecordsLast10Years` with median values.

# In[40]:


data[data.PublicRecordsLast10Years.isna()].sample(5)


# In[41]:


median_10_public_records = data["PublicRecordsLast10Years"].median()
data["PublicRecordsLast10Years"].fillna(median_10_public_records, inplace=True)
data["PublicRecordsLast10Years"].isnull().sum()


# I will impute missing values in `PublicRecordsLast12Months` with median values.

# In[42]:


data[data.PublicRecordsLast12Months.isna()].sample(5)


# In[43]:


median_public_records_12 = data["PublicRecordsLast12Months"].median()
data["PublicRecordsLast12Months"].fillna(median_public_records_12, inplace=True)
data["PublicRecordsLast12Months"].isnull().sum()


# I will impute missing values in `RevolvingCreditBalance` with median values.

# In[44]:


data[data.RevolvingCreditBalance.isna()].sample(5)


# In[45]:


median_revolving_credit = data["RevolvingCreditBalance"].median()
data["RevolvingCreditBalance"].fillna(median_revolving_credit, inplace=True)
data["RevolvingCreditBalance"].isnull().sum()


# I will impute missing values in `BankcardUtilization` with median values.

# In[46]:


data[data.BankcardUtilization.isna()].sample(5)


# In[47]:


median_bankcard_utilization = data["BankcardUtilization"].median()
data["BankcardUtilization"].fillna(median_bankcard_utilization, inplace=True)
data["BankcardUtilization"].isnull().sum()


# I will impute missing values in `AvailableBankcardCredit` with median values.

# In[48]:


data[data.AvailableBankcardCredit.isna()].sample(5)


# In[49]:


median_available_bankcard = data["AvailableBankcardCredit"].median()
data["AvailableBankcardCredit"].fillna(median_available_bankcard, inplace=True)
data["AvailableBankcardCredit"].isnull().sum()


# I will impute missing values in `TotalTrades` with median values.

# In[50]:


data[data.TotalTrades.isna()].sample(5)


# In[51]:


median_total_trades = data["TotalTrades"].median()
data["TotalTrades"].fillna(median_total_trades, inplace=True)
data["TotalTrades"].isnull().sum()


# I will impute missing values in `TradesNeverDelinquent.per` with median values.

# In[52]:


data[data["TradesNeverDelinquent.per"].isna()].sample(5)


# In[53]:


median_never_delinquent = data["TradesNeverDelinquent.per"].median()
data["TradesNeverDelinquent.per"].fillna(median_never_delinquent, inplace=True)
data["TradesNeverDelinquent.per"].isnull().sum()


# I will impute missing values in `TradesOpenedLast6Months` with median values.

# In[54]:


data[data.TradesOpenedLast6Months.isna()].sample(5)


# In[55]:


median_6_trades_opened = data["TradesOpenedLast6Months"].median()
data["TradesOpenedLast6Months"].fillna(median_6_trades_opened, inplace=True)
data["TradesOpenedLast6Months"].isnull().sum()


# I will impute missing values in `DebtToIncomeRatio` with median values.

# In[56]:


data[data.DebtToIncomeRatio.isna()].sample(5)


# In[57]:


median_debt_to_income = data["DebtToIncomeRatio"].median()
data["DebtToIncomeRatio"].fillna(median_debt_to_income, inplace=True)
data["DebtToIncomeRatio"].isnull().sum()


# I will impute missing values in `TotalProsperLoans`, and the rest of the features below, with 0, as in this case it appears that in the original dataset, this feature had a value only when a Prosper loan had already been taken out.

# In[58]:


data[data.TotalProsperLoans.isna()].sample(5)
     
# TotalProsperPaymentsBilled             10539 non-null float64
# OnTimeProsperPayments                  10539 non-null float64
# ProsperPaymentsLessThanOneMonthLate    10539 non-null float64
# ProsperPaymentsOneMonthPlusLate        10539 non-null float64
# ProsperPrincipalBorrowed               10539 non-null float64
# ProsperPrincipalOutstanding            10539 non-null float64


# In[59]:


data["TotalProsperLoans"].fillna(0, inplace=True)
print(data["TotalProsperLoans"].isnull().sum())

data["TotalProsperPaymentsBilled"].fillna(0, inplace=True)
print(data["TotalProsperPaymentsBilled"].isnull().sum())

data["OnTimeProsperPayments"].fillna(0, inplace=True)
print(data["OnTimeProsperPayments"].isnull().sum())

data["ProsperPaymentsLessThanOneMonthLate"].fillna(0, inplace=True)
print(data["ProsperPaymentsLessThanOneMonthLate"].isnull().sum())

data["ProsperPaymentsOneMonthPlusLate"].fillna(0, inplace=True)
print(data["ProsperPaymentsOneMonthPlusLate"].isnull().sum())

data["ProsperPrincipalBorrowed"].fillna(0, inplace=True)
print(data["ProsperPrincipalBorrowed"].isnull().sum())

data["ProsperPrincipalOutstanding"].fillna(0, inplace=True)
print(data["ProsperPrincipalOutstanding"].isnull().sum())


# I will impute missing values in `ScorexChangeAtTimeOfListing` with median values.

# In[60]:


data[data.ScorexChangeAtTimeOfListing.isna()].sample(5)


# In[61]:


median_scorex = data["ScorexChangeAtTimeOfListing"].median()
data["ScorexChangeAtTimeOfListing"].fillna(median_scorex, inplace=True)
data["ScorexChangeAtTimeOfListing"].isnull().sum()


# I will impute missing values in `Rating` with the label `Unknown`.

# In[62]:


data[data.Rating.isna()].sample(5)


# In[63]:


data["Rating"].fillna("Unknown", inplace=True)
data["Rating"].isnull().sum()


# I will impute missing values in `CreditScore` with median values.

# In[64]:


data[data.CreditScore.isna()].sample(5)


# In[65]:


median_credit_score = data["CreditScore"].median()
data["CreditScore"].fillna(median_credit_score, inplace=True)
data["CreditScore"].isnull().sum()


# I will impute missing values in `CreditHistoryAge` with median values.

# In[66]:


data[data.CreditHistoryAge.isna()].sample(5)


# In[67]:


median_credit_age = data["CreditHistoryAge"].median()
data["CreditHistoryAge"].fillna(median_credit_age, inplace=True)
data["CreditHistoryAge"].isnull().sum()


# I will now remove the listing and member keys, as they are of no further use (as far as I can see).

# In[68]:


data.drop(["ListingKey","MemberKey"], axis=1, inplace=True)


# ### Data Exploration
# 
# I will now explore the data further, and fix or note any obvious issues I encounter.

# In[69]:


data.describe()


# `ListingCategory` is currently a numerical variable, which makes it difficult to interpret without repeatedly referring to the variable definition sheet; I therefore change the numbers to corresponding labels provided with the data. (cf. https://rstudio-pubs-static.s3.amazonaws.com/86324_ab1e2e2fa210452f80a1c6a1476d7a2a.html)

# In[70]:


data.replace(to_replace={"ListingCategory.num": 
                         {0: "Unknown", 1: "Debt", 2: "Reno", 3: "Business", 4: "Personal", 5: "Student", 
                          6: "Auto", 7: "Other", 8: "Baby", 9: "Boat", 10: "Cosmetic", 11: "Engagement", 
                          12: "Green", 13: "Household", 14: "LargePurchase", 15: "Medical", 16: "Motorcycle", 
                          17: "RV", 18: "Taxes", 19: "Vacation", 20: "Wedding"}}, inplace=True)

data.rename(columns={"ListingCategory.num": "ListingCategory"}, inplace=True, index=str)

data.ListingCategory.sample(10)


# In[71]:


data.info()


# There appear to be no more missing values, except in those columns where values principally could not be imputed without further analysis.

# In[72]:


# total number of records
n_records = data.shape[0]

# number of records where loan was completed
n_completed = data[data["Completed.num"]==1].shape[0]

# number of records where loan was not completed
n_not_completed = data[data["Completed.num"]==0].shape[0]

# percent of loans completed
percent_completed = n_completed/n_records * 100

print("Total number of records: {}".format(n_records))
print("Historical loans completed: {}".format(n_completed))
print("Historical loans not completed: {}".format(n_not_completed))
print("Percent of historical loans completed: {}%".format(round(percent_completed, 2)))


# This dataset is imbalanced; significantly more loans were completed (repaid) than not.  This may have consequences for choice of algorithm.

# ## Preparing the Data

# ### Transforming Skewed Continuous Features
# 
# Below, I plot continuous features in the dataset, and see if they show any significant skew.  If so, I add them to a list of features to be log-transformed, which reduces the range of values caused by outliers, and makes it less likely that outliers will significantly influence any subsequent analyses.  Further, some methods depend on feature normality.

# In[73]:


data.sample(15)


# In[74]:


# Split the data into features and target label
completion = data["Completed.num"]
default = 1 - completion
features_raw = data.drop("Completed.num", axis = 1)

# Visualize continuous features of original data
continuous_features = ["Term", "BorrowerRate", "LenderYield", "EstimatedEffectiveYield", 
                       "EstimatedLoss", "EstimatedReturn", "ProsperScore", "EmploymentStatusDuration", 
                       "CurrentCreditLines", "OpenCreditLines", "TotalCreditLinespast7years", 
                       "OpenRevolvingAccounts", "OpenRevolvingMonthlyPayment", "InquiriesLast6Months",
                       "TotalInquiries", "CurrentDelinquencies", "AmountDelinquent", "DelinquenciesLast7Years",
                       "PublicRecordsLast10Years", "PublicRecordsLast12Months", "RevolvingCreditBalance",
                       "BankcardUtilization", "AvailableBankcardCredit", "TotalTrades", 
                       "TradesNeverDelinquent.per", "TradesOpenedLast6Months", "DebtToIncomeRatio", 
                       "StatedMonthlyIncome", "TotalProsperLoans", "TotalProsperPaymentsBilled", 
                       "OnTimeProsperPayments", "ProsperPaymentsLessThanOneMonthLate", 
                       "ProsperPaymentsOneMonthPlusLate", "ProsperPrincipalBorrowed", 
                       "ProsperPrincipalOutstanding", "ScorexChangeAtTimeOfListing", "LoanOriginalAmount",
                       "MonthlyLoanPayment", "Recommendations", "InvestmentFromFriendsCount", 
                       "InvestmentFromFriendsAmount", "Investors", "CreditScore", "CreditHistoryAge"]

for feature in continuous_features:
    size, scale = 1000, 10
    data[feature].plot.hist(grid=True, bins=20, rwidth=0.9, color='#607c8e')
    plt.title(feature)
    plt.xlabel('Counts')
    plt.ylabel(feature)
    plt.grid(axis='y', alpha=0.75)
    plt.show()


# Below, I perform a log-transformation on those features I noted to be skewed, adding a small number to `x`, since the logarithm of 0 is undefined.

# In[75]:


# Log-transform the skewed features
skewed = ['EstimatedEffectiveYield', 'EstimatedLoss', 'EmploymentStatusDuration', 'CurrentCreditLines',
          'OpenCreditLines', 'TotalCreditLinespast7years', 'OpenRevolvingAccounts', 'OpenRevolvingMonthlyPayment',
          'InquiriesLast6Months', 'TotalInquiries', 'CurrentDelinquencies', 'AmountDelinquent', 
          'DelinquenciesLast7Years', 'PublicRecordsLast10Years', 'PublicRecordsLast12Months',
          'RevolvingCreditBalance', 'BankcardUtilization', 'AvailableBankcardCredit', 'TotalTrades',
          'TradesNeverDelinquent.per', 'TradesOpenedLast6Months', 'DebtToIncomeRatio', 'StatedMonthlyIncome',
          'TotalProsperLoans', 'TotalProsperPaymentsBilled', 'OnTimeProsperPayments', 
          'ProsperPaymentsLessThanOneMonthLate', 'ProsperPaymentsOneMonthPlusLate', 'ProsperPrincipalBorrowed',
          'ProsperPrincipalOutstanding', 'LoanOriginalAmount', 'MonthlyLoanPayment', 'Recommendations',
          'InvestmentFromFriendsCount', 'InvestmentFromFriendsAmount', 'Investors', 'CreditHistoryAge']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 0.1))

# Visualize the new log distributions
for feature in skewed:
    size, scale = 1000, 10
    features_log_transformed[feature].plot.hist(grid=True, bins=20, rwidth=0.9,
                   color='#607c8e')
    plt.title(feature)
    plt.xlabel('Counts')
    plt.ylabel(feature)
    plt.grid(axis='y', alpha=0.75)
    plt.show()


# Many of the features are still not normally distributed, but their values are by and large less extreme, and less likely to affect learning algorithms.  In the future, I will explore other transformations that may be applied to non-normally-distributed features.

# ### Coding Categorical Data
# 
# Since many learning algorithms only take numerical input, below I dummy-code categorical variables, with `0` indicating the absence of a feature/feature category, and `1` indicating its presence.

# In[76]:


features_log_transformed.sample(10)


# Below, I dummy-code all categorical features:

# In[77]:


# dummy-code the log-transformed features
features_final = pd.get_dummies(features_log_transformed)

# Print the number of features after dummy-coding
encoded = list(features_final.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))

print(encoded)


# ## Dimensionality Reduction

# In this section, I use principal component analysis (PCA) to see if the features in my dataset can be reduced to fewer dimensions, which may improve algorithm performance, and reduce feature redundancy.

# Below, I split the data into training and testing datasets.

# In[78]:


from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

X_train, X_test, y_train, y_test = train_test_split(features_final, completion, test_size=0.15, random_state=42)


# ### Normalizing Numerical Features
# 
# Prior to using PCA, I scale all numerical features, which ensures that algorithms treat features equally, rather than giving some undue weight.

# In[79]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler() # default=(0, 1)

X_train_scaled = X_train
X_train_scaled[continuous_features] = scaler.fit_transform(X_train[continuous_features])

X_test_scaled = X_test
X_test_scaled[continuous_features] = scaler.fit_transform(X_test[continuous_features])


# In[80]:


# Show the results of the split
print("Training set has {} samples.".format(X_train_scaled.shape[0]))
print("Testing set has {} samples.".format(X_test_scaled.shape[0]))


# Below, I drop all columns with remaining missing values, and save the data to a separate testing/training set which can be used with algorithms which handle missing data natively.

# In[81]:


X_test_scaled.sample(10)


# In[82]:


X_train_non_na = X_train_scaled.drop(['EstimatedEffectiveYield', 'EstimatedLoss',
                                      'EstimatedReturn', 'ProsperScore'], axis=1)
X_test_non_na = X_test_scaled.drop(['EstimatedEffectiveYield', 'EstimatedLoss',
                                    'EstimatedReturn', 'ProsperScore'], axis=1)

X_train_full = X_train_scaled
X_test_full = X_test_scaled


# ### Applying Dimensionality Reduction
# 
# Below, I reduce the features to 10 components, and then 3 components.  I further reduce the dataset to the top 30% of most important features (those that account for the most variance), and to the top 10%.  All are saved to separate testing/training sets.

# In[83]:


from sklearn.decomposition import PCA

pca = PCA(n_components=10, random_state=42)
pca.fit(X_train_non_na)

X_train_pca10 = pca.transform(X_train_non_na)
X_test_pca10 = pca.transform(X_test_non_na)

pca.explained_variance_ratio_


# In[84]:


pca = PCA(n_components=3, random_state=42)
pca.fit(X_train_non_na)

X_train_pca3 = pca.transform(X_train_non_na)
X_test_pca3 = pca.transform(X_test_non_na)

pca.explained_variance_ratio_


# In[85]:


from sklearn.feature_selection import SelectPercentile

X_train_reduce30 = SelectPercentile(percentile=30).fit_transform(X_train_non_na, y_train)
X_test_reduce30 = SelectPercentile(percentile=30).fit_transform(X_test_non_na, y_test)

X_train_reduce10 = SelectPercentile(percentile=10).fit_transform(X_train_non_na, y_train)
X_test_reduce10 = SelectPercentile(percentile=10).fit_transform(X_test_non_na, y_test)


# Above, it looks like the most important three features account for quite a bit of the variance, and the rest account for comparatively tiny portions.  This may make the use of a highly reduced dataset feasible.

# ----
# ## Evaluating Model Performance
# 

# ### The Naive Predictor

# Below, we look at how a simple model, which always predicts that a loan is repaid, performs.  Although the accuracy and F-score in this case is rather high, it is clear that this model performs rather poorly.  An improved model must have higher precision -- which denotes how many of the loans identified as repaid were, in fact, repaid -- even if at the expense of recall, particularly given that lenders should be more wary of investing in loans unlikely to pay off, even if this involves risk of not investing in potentially high-yield loans.

# In[86]:


# true positives
TP = np.sum(completion)
# false positives
FP = completion.count() - TP

# true negatives
TN = 0
# false negatives
FN = 0

# accuracy, precision and recall
accuracy = TP/(TP+FP)
recall = TP/(TP+FN)
precision = TP/(TP+FP)

# F-score for beta = 1
fscore = (1+1**2) * (precision * recall)/(1**2 * precision + recall)

# Print the results 
print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}, Precision: {:.4f}, Recall: {:.4f}]".format(accuracy, fscore, precision, recall))


# ## Model Application
# 

# I picked three Ensemble methods commonly used for the task of classification - one averaging method (random forests), and two boosting methods.  I also picked Logistic Regression, as having an output of probabilities allows for more interpretability, and flexibility after the fact in setting thresholds for decisions.  I broadly retain some of my previous descriptions of these model types, where I have previously used them.
# 
# For strengths and weaknesses of the relevant models, I primarily consulted the following sources, keeping in mind that heuristics do not necessarily apply to each data set:
# 
# https://medium.com/@randylaosat/machine-learning-whats-inside-the-box-861f5c7e72a3
# 
# https://medium.com/@vijaya.beeravalli/comparison-of-machine-learning-classification-models-for-credit-card-default-data-c3cf805c9a5a
# 
# https://www.dummies.com/programming/big-data/data-science/machine-learning-dummies-cheat-sheet/
# 
# https://hackernoon.com/boosting-algorithms-adaboost-gradient-boosting-and-xgboost-f74991cad38c
# 
# https://medium.com/@grohith327/gradient-boosting-and-xgboost-90862daa6c77
# 
# #### Ensemble Methods: Random Forest
# 
# The strengths of this model, which averages across multiple decision trees, are the following: it natively handles categorical variables; it is less prone to overfitting than a single decision tree -- therefore, it is more likely to select relevant features; and it frequently outperforms other methods on accuracy.  **These features make this model a good candidate for this problem, which has numerous categorical and continuous features.  Given that after one-hot conversion, there were 88 total features, a model which will automatically select the most important features is of particular importance.**  Additionally, it can handle imbalanced data, which applies to this dataset, and is a flexible algorithm that does not require a lot of parameter tuning.
# 
# The weaknesses of this model are that it doesn't tend to perform well with a bad set of features; it's not very transparent, and it's hard to interpret what's going on in the algorithm. Further, too many trees can slow down the algorithm.
# 
# #### Ensemble Methods: AdaBoost
# 
# AdaBoost is particularly well-suited for boosting the performance of decision trees on binary classification tasks.  It typically does not overfit despite excellent accuracy (although it is to date unclear exactly how), and frequently outperforms other methods.  **These features make it a good model to attempt for this problem, given that the algorithm can be used to boost the performance of classifiers that work well natively with categorical data, such as decision trees, and like random forests will automatically select those features that are most important.**  It additionally does not require variable transformation, and has relatively few parameters that need tweaking.
# 
# The weaknesses of this model are that it does not deal well with noisy data, and the efficiency of the algorithm is affected by outliers, since the algorithm attempts to fit each point.
# 
# #### Logistic Regression
# 
# The strengths of this model are that it's fairly easy to interpret in terms of probabilities; relatively unlikely to overfit; fast; well-suited for binary classification tasks; and explanatory variables can be continuous or categorical.  **These features, and in particular the fact that results are probabilities, make this model a good candidate for this problem, since probabilities allow for post-hoc adjustment of the threshold for whether a lender should fund a particular loan, perhaps depending on their personal finances or risk aversion.**
# 
# The downsides of this model are that it's not particularly good at capturing complex or non-linear relationships between features, or dealing with multiple/non-linear decision boundaries; generally, it's not very flexible.
# 
# #### Ensemble Methods: XGBoost
# 
# XGBoost, recommended by a reviewer, is highly robust to irregularities in data, and like AdaBoost, is a boosting algorithm which tries to create a strong classifier from a series of weaker classifiers.  It additionally natively handles missing data.  **Given that the dataset I have is quite noisy, with quite a bit of missing or imputed data, and is particularly suited to decision trees, this algorithm would be appropriate to try.**

# ### Training and Predicting Pipeline
# 
# Below I use a training function adapted from another Udacity project to evaluate the peformance of the various classifiers, with repect to time and various metrics, on testing and training data.

# In[87]:


from sklearn.metrics import fbeta_score, accuracy_score, f1_score, precision_score, recall_score

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    start = time() # Get start time
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() # Get end time
    
    results['train_time'] = end - start
        
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time
    
    results['pred_time'] = end - start
            
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train[:300])
        
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    results['f_train'] = f1_score(y_train[:300], predictions_train[:300])
        
    results['f_test'] = f1_score(y_test, predictions_test)
    
    results['p_train'] = precision_score(y_train[:300], predictions_train[:300])
        
    results['p_test'] = precision_score(y_test, predictions_test)
    
    results['r_train'] = recall_score(y_train[:300], predictions_train[:300])
        
    results['r_test'] = recall_score(y_test, predictions_test)
       
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    return results


# ### Model Evaluation
# 

# #### Full dataset without missing data

# In[88]:


from time import time

clf_A = RandomForestClassifier(random_state = 42)
clf_B = AdaBoostClassifier(random_state = 42)
clf_C = LogisticRegression(random_state = 42)
clf_D = XGBClassifier(random_state = 42)

samples_100 = len(y_train)
samples_10 = int(samples_100 * 0.1)
samples_1 = int(samples_100 * 0.01)

results_non_na = {}
for clf in [clf_A, clf_B, clf_C, clf_D]:
    clf_name = clf.__class__.__name__
    results_non_na[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results_non_na[clf_name][i] =         train_predict(clf, samples, X_train_non_na, y_train, X_test_non_na, y_test)


# In[89]:


vs.evaluate(results_non_na, accuracy, precision, fscore, "Performance Metrics: Missing Data Removed")


# #### PCA: 10 most important features

# In[90]:


clf_A = RandomForestClassifier(random_state = 42)
clf_B = AdaBoostClassifier(random_state = 42)
clf_C = LogisticRegression(random_state = 42)
clf_D = XGBClassifier(random_state = 42)

samples_100 = len(y_train)
samples_10 = int(samples_100 * 0.1)
samples_1 = int(samples_100 * 0.01)

results_pca10 = {}
for clf in [clf_A, clf_B, clf_C, clf_D]:
    clf_name = clf.__class__.__name__
    results_pca10[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results_pca10[clf_name][i] =         train_predict(clf, samples, X_train_pca10, y_train, X_test_pca10, y_test)


# In[91]:


vs.evaluate(results_pca10, accuracy, precision, fscore, "Performance Metrics: 10 most important features (PCA)")


# #### PCA: 3 most important features

# In[92]:


clf_A = RandomForestClassifier(random_state = 42)
clf_B = AdaBoostClassifier(random_state = 42)
clf_C = LogisticRegression(random_state = 42)
clf_D = XGBClassifier(random_state = 42)

samples_100 = len(y_train)
samples_10 = int(samples_100 * 0.1)
samples_1 = int(samples_100 * 0.01)

results_pca3 = {}
for clf in [clf_A, clf_B, clf_C, clf_D]:
    clf_name = clf.__class__.__name__
    results_pca3[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results_pca3[clf_name][i] =         train_predict(clf, samples, X_train_pca3, y_train, X_test_pca3, y_test)


# In[93]:


vs.evaluate(results_pca3, accuracy, precision, fscore, "Performance Metrics: 3 most important features (PCA)")


# #### PCA: top 30% of features

# In[94]:


clf_A = RandomForestClassifier(random_state = 42)
clf_B = AdaBoostClassifier(random_state = 42)
clf_C = LogisticRegression(random_state = 42)
clf_D = XGBClassifier(random_state = 42)

samples_100 = len(y_train)
samples_10 = int(samples_100 * 0.1)
samples_1 = int(samples_100 * 0.01)

results_reduce30 = {}
for clf in [clf_A, clf_B, clf_C, clf_D]:
    clf_name = clf.__class__.__name__
    results_reduce30[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results_reduce30[clf_name][i] =         train_predict(clf, samples, X_train_reduce30, y_train, X_test_reduce30, y_test)


# In[95]:


vs.evaluate(results_reduce30, accuracy, precision, fscore, "Performance Metrics: Top 30% features (PCA)")


# #### PCA: top 10% of features

# In[96]:


clf_A = RandomForestClassifier(random_state = 42)
clf_B = AdaBoostClassifier(random_state = 42)
clf_C = LogisticRegression(random_state = 42)
clf_D = XGBClassifier(random_state = 42)

samples_100 = len(y_train)
samples_10 = int(samples_100 * 0.1)
samples_1 = int(samples_100 * 0.01)

results_reduce10 = {}
for clf in [clf_A, clf_B, clf_C, clf_D]:
    clf_name = clf.__class__.__name__
    results_reduce10[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results_reduce10[clf_name][i] =         train_predict(clf, samples, X_train_reduce10, y_train, X_test_reduce10, y_test)


# In[97]:


vs.evaluate(results_reduce10, accuracy, precision, fscore, "Performance Metrics: Top 10% features (PCA)")


# #### Full dataset with missing data (XGBoost only, which unlike AdaBoost handles missing values)

# In[98]:


clf_A = XGBClassifier(random_state = 42)

samples_100 = len(y_train)
samples_10 = int(samples_100 * 0.1)
samples_1 = int(samples_100 * 0.01)

results_full = {}
for clf in [clf_A]:
    clf_name = clf.__class__.__name__
    results_full[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results_full[clf_name][i] =         train_predict(clf, samples, X_train_full, y_train, X_test_full, y_test)


# In[99]:


results_spliced = results_non_na
results_spliced["XGBClassifier_full"] = results_full["XGBClassifier"]


# In[100]:


vs.evaluate5(results_spliced, accuracy, precision, fscore, "Performance Metrics: Missing data removed vs. XGBoost using full data set")


# ----
# ## Optimizing Results

# **As can be seen, the Random Forest model tends to overfit the training data, and then perform similar to other models on precision in testing, or underperform significantly on other measures.  It therefore appears comparatively unsuited to this task.  Logistic Regression tends to perform similarly to other models or underperform on precision, and performs moderately to significantly worse on other measures.  It rarely performs better than the naive classifier, and when it does, its performance is similar to that of other methods.  It therefore appears similarly unsuited, and its performance suffers significantly when the features are reduced to several dimensions.  In contrast, the boosting methods have a less pronounced tendency to overfit the data than the Random Forest method, show similar performance on precision, and higher performance on other measures (accuracy and f-score).  Although they are slower to train, they appear to be roughly comparable in testing, with XGBoost having an edge over AdaBoost.**
# 
# **AdaBoost and XGBoost perform roughly similarly on the testing and training data, with XGBoost perhaps having a slight edge in testing (no improvement on measures is seen from including the variables with missing data, in the latter case).  Performance appears to be best all-around (precision higher than naive; F1 metric no lower than naive) when all available features are used.  Both algorithms are similarly well-suited for the large number of continuous and categorical variables, as seen in this dataset, given that ensemble methods will pick out the most important features, and decision tree-based classifiers are paticularly well-suited to categorical data.  It is difficult to choose between the two methods based on the above results; I will therefore attempt to optimize both models.  Below, I briefly discuss how the two methods work.**

# ### AdaBoost

# **AdaBoost takes a family of weak classifiers, which may perform only slightly better than chance at correctly classifying any given point, combines them, and lets them 'vote' on the correct category of any given point.  When a large number of such weak classifiers vote on any given point, the chance that this point will be correctly classified increases significantly.  By default, AdaBoost uses decision tree classifiers to classify points, although it can use different base classifiers.  The process of classifying points continues either until each point is correctly classified in a labeled data set, or until the maximum number of iterations specified is reached.**
# 
# **The image below illustrates this: a weak classifier in Box 1 classifies several of the positive points correctly, but misclassifies the rest of the positive points.  The weights of the misclassified samples are then increased, and the next weak classifier (Box 2) is more likely to classify these points correctly.  The weights of misclassified samples is increased again, and the process repeats.  In the end, the weak classifiers are combined, and 'vote' on the category a given point belongs to:**
# 
# ![Boosting](boosting.png)
# 
# **The drawback of this method, as mentioned, is that it does not deal well with noisy data, given that it attempts to categorize all points correctly.  It is not, however, prone to overfitting, although the reason for this is not clear.**
# 
# Resources:
# 
# https://hackernoon.com/boosting-algorithms-adaboost-gradient-boosting-and-xgboost-f74991cad38c (image copied from this resource)

# ### XGBoost
# 
# **XGBoost is a gradient boosting algorithm, which like AdaBoost attempts to create a strong learner from a group of weak learners, giving increased weight to previously misclassified outcomes, and lower weight to correctly classified outcomes.  Where it differs is that in gradient boosting, new learners work to reduce the residuals of previous learners, thus minimizing loss.  It further penalizes excessive complexity in decision trees, and has a randomization parameter which can reduce the correlation between trees.**
# 
# **The process of iteratively reducing residuals can be thus visualized, with error virtually eliminated by the 20th iteration:**
# 
# ![XGB1](xgb1.png)
# 
# ![XGB2](xgb2.png)
# 
# ![XGB3](xgb3.png)
# 
# Resources:
# 
# https://hackernoon.com/gradient-boosting-and-xgboost-90862daa6c77 (remaining images copied from this source)
# https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/
# https://www.datacamp.com/community/tutorials/xgboost-in-python
# https://medium.com/syncedreview/tree-boosting-with-xgboost-why-does-xgboost-win-every-machine-learning-competition-ca8034c0b283

# ### Model Tuning (AdaBoost)

# In[101]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

clf = AdaBoostClassifier(random_state=42)

parameters = dict(n_estimators=[10, 50, 100, 500, 1000], 
                  learning_rate=[0.01, 0.1, 0.5, 1],
                  algorithm=['SAMME', 'SAMME.R'])

scorer = {'F1': make_scorer(f1_score), 
          'Precision': make_scorer(precision_score), 
          'Recall': make_scorer(recall_score)}

grid_obj = GridSearchCV(clf, param_grid = parameters, scoring=scorer, refit='Precision', verbose=5)

grid_fit = grid_obj.fit(X_train_non_na, y_train)

best_adaboost_clf = grid_fit.best_estimator_

adaboost_predictions = (clf.fit(X_train_non_na, y_train)).predict(X_test_non_na)
best_adaboost_predictions = best_adaboost_clf.predict(X_test_non_na)


# In[ ]:


dill.dump_session('notebook_env1.db')


# In[102]:


best_adaboost_clf.get_params()


# In[103]:


print("Unoptimized Model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, adaboost_predictions)))
print("F-score on testing data: {:.4f}".format(f1_score(y_test, adaboost_predictions)))
print("Precision on testing data: {:.4f}".format(precision_score(y_test, adaboost_predictions)))
print("Recall on testing data: {:.4f}".format(recall_score(y_test, adaboost_predictions)))
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_adaboost_predictions)))
print("Final F-score on the testing data: {:.4f}".format(f1_score(y_test, best_adaboost_predictions)))
print("Final precision on the testing data: {:.4f}".format(precision_score(y_test, best_adaboost_predictions)))
print("Final recall on the testing data: {:.4f}".format(recall_score(y_test, best_adaboost_predictions)))


# ### Model Tuning (XGBoost)

# In[109]:


clf = XGBClassifier(random_state=42)

parameters = dict(n_estimators=[10, 50, 100, 500, 1000], 
                  learning_rate=[0.01, 0.1, 0.5, 1], 
                  max_depth=[3,6,9])

scorer = {'F1': make_scorer(f1_score), 
          'Precision': make_scorer(precision_score), 
          'Recall': make_scorer(recall_score)}

grid_obj = GridSearchCV(clf, param_grid = parameters, scoring=scorer, refit='Precision', verbose=5)

grid_fit = grid_obj.fit(X_train_non_na, y_train)

best_xgboost_clf = grid_fit.best_estimator_

xgboost_predictions = (clf.fit(X_train_non_na, y_train)).predict(X_test_non_na)
best_xgboost_predictions = best_xgboost_clf.predict(X_test_non_na)


# In[ ]:


best_xgboost_clf.get_params()


# In[ ]:


print("Unoptimized Model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, xgboost_predictions)))
print("F-score on testing data: {:.4f}".format(f1_score(y_test, xgboost_predictions)))
print("Precision on testing data: {:.4f}".format(precision_score(y_test, xgboost_predictions)))
print("Recall on testing data: {:.4f}".format(recall_score(y_test, xgboost_predictions)))
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_xgboost_predictions)))
print("Final F-score on the testing data: {:.4f}".format(f1_score(y_test, best_xgboost_predictions)))
print("Final precision on the testing data: {:.4f}".format(precision_score(y_test, best_xgboost_predictions)))
print("Final recall on the testing data: {:.4f}".format(recall_score(y_test, best_xgboost_predictions)))


# ### Final Model Evaluation
# 
# 

# #### Results:
# 
# |     Metric     | Naive Predictor   | Unoptimized Model | Optimized Model |
# | :------------: | :---------------: | :---------------: | :-------------: | 
# | Accuracy Score | 0.6912            | 0.7217            |   0.6898        |
# | F-score        | 0.8174            | 0.8051            |   0.7609        |
# | Precision      | 0.6912            | 0.7725            |   0.8047        |
# | Recall         | 1.0000            | 0.8406            |   0.7217        |

# **Here it can be seen that although accuracy and recall are lower in the optimized model than in the unoptimized model, or the naive predictor, precision is higher, which should assist lenders in avoiding those loans most likely to default.**

# ----
# ## Feature Importance

# ### Extracting Feature Importance
# 
# Below, I plot the most important features for the optimal model chosen above.  As can be seen, the magnitude of the monthly loan payment, the age of credit history, the amoung of revolving credit balance, the loan borrower rate, and the loan origination date are the best predictors of whether a loan will default, or not.  The first four predictors tend to correlate with a borrower's credit worthiness, and it is unsurprising that they primarily determine whether a loan will be paid back, or not.  In contrast, it is unclear why loan origination date should be a good predictor of a loan defaulting -- however, I have previously found that loan completion and lender profit tend to 'cycle' over time, for reasons that would need to be expored further:
# 
# https://eskrav.github.io/udacity-data-analyst/explore-and-summarize/explore-and-summarize.html#lender_profit_by_loan_origination_quarter

# In[ ]:


model = best_adaboost_clf.fit(X_train_non_na, y_train)

importances_adaboost = best_adaboost_clf.feature_importances_

vs.feature_plot(importances_adaboost, X_train_non_na, y_train)


# In[ ]:


model = best_xgboost_clf.fit(X_train_non_na, y_train)

importances_xgboost = best_xgboost_clf.feature_importances_

vs.feature_plot(importances_xgboost, X_train_non_na, y_train)


# ### Feature Selection
# 
# Here, I look at how a model performs if only the five most important features are used.  Accuracy, f-score, and recall suffer, but predicion here arguably the most important metric, suffers only marginally.  With further exploration, it is possible that a model utilising only the most important features would show adequate performance in production.

# In[ ]:


# Import functionality for cloning a model
from sklearn.base import clone

# Reduce the feature space
X_train_reduced = X_train_non_na[X_train_non_na.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test_non_na[X_test_non_na.columns.values[(np.argsort(importances)[::-1])[:5]]]

# Train on the "best" model found from grid search earlier
adaboost_clf = (clone(best_adaboost_clf)).fit(X_train_reduced, y_train)

# Make new predictions
reduced_adaboost_predictions = adaboost_clf.predict(X_test_reduced)

# Report scores from the final model using both versions of data
print("Final Model trained on full data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_adaboost_predictions)))
print("F-score on testing data: {:.4f}".format(f1_score(y_test, best_adaboost_predictions)))
print("Precision on testing data: {:.4f}".format(precision_score(y_test, best_adaboost_predictions)))
print("Recall on testing data: {:.4f}".format(recall_score(y_test, best_adaboost_predictions)))
print("\nFinal Model trained on reduced data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_adaboost_predictions)))
print("F-score on testing data: {:.4f}".format(f1_score(y_test, reduced_adaboost_predictions)))
print("Precision on testing data: {:.4f}".format(precision_score(y_test, reduced_adaboost_predictions)))
print("Recall on testing data: {:.4f}".format(recall_score(y_test, reduced_adaboost_predictions)))


# In[ ]:


# Train on the "best" model found from grid search earlier
xgboost_clf = (clone(best_xgboost_clf)).fit(X_train_reduced, y_train)

# Make new predictions
reduced_xgboost_predictions = xgboost_clf.predict(X_test_reduced)

# Report scores from the final model using both versions of data
print("Final Model trained on full data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_xgboost_predictions)))
print("F-score on testing data: {:.4f}".format(f1_score(y_test, best_xgboost_predictions)))
print("Precision on testing data: {:.4f}".format(precision_score(y_test, best_xgboost_predictions)))
print("Recall on testing data: {:.4f}".format(recall_score(y_test, best_xgboost_predictions)))
print("\nFinal Model trained on reduced data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_xgboost_predictions)))
print("F-score on testing data: {:.4f}".format(f1_score(y_test, reduced_xgboost_predictions)))
print("Precision on testing data: {:.4f}".format(precision_score(y_test, reduced_xgboost_predictions)))
print("Recall on testing data: {:.4f}".format(recall_score(y_test, reduced_xgboost_predictions)))


# In[ ]:


from xgboost import plot_tree

plot_tree(best_xgboost_clf, rankdir='LR')
plt.show()


# In[ ]:


dill.dump_session('notebook_env2.db')

