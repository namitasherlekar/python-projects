#!/usr/bin/env python
# coding: utf-8

# # Trade&Ahead Project
# 
# ## Context
# 
# The stock market has consistently proven to be a good place to invest in and save for the future. There are a lot of compelling reasons to invest in stocks. It can help in fighting inflation, create wealth, and also provides some tax benefits. Good steady returns on investments over a long period of time can also grow a lot more than seems possible. Also, thanks to the power of compound interest, the earlier one starts investing, the larger the corpus one can have for retirement. Overall, investing in stocks can help meet life's financial aspirations.
# 
# It is important to maintain a diversified portfolio when investing in stocks in order to maximise earnings under any market condition. Having a diversified portfolio tends to yield higher returns and face lower risk by tempering potential losses when the market is down. It is often easy to get lost in a sea of financial metrics to analyze while determining the worth of a stock, and doing the same for a multitude of stocks to identify the right picks for an individual can be a tedious task. By doing a cluster analysis, one can identify stocks that exhibit similar characteristics and ones which exhibit minimum correlation. This will help investors better analyze stocks across different market segments and help protect against risks that could make the portfolio vulnerable to losses.
# 
# 
# ## Objective
# 
# Trade&Ahead is a financial consultancy firm who provide their customers with personalized investment strategies. They have hired you as a Data Scientist and provided you with data comprising stock price and some financial indicators for a few companies listed under the New York Stock Exchange. They have assigned you the tasks of analyzing the data, grouping the stocks based on the attributes provided, and sharing insights about the characteristics of each group.
# 
# ### Data Dictionary
# 
# - Ticker Symbol: An abbreviation used to uniquely identify publicly traded shares of a particular stock on a particular stock market
# - Company: Name of the company
# - GICS Sector: The specific economic sector assigned to a company by the Global Industry Classification Standard (GICS) that best defines its business operations
# - GICS Sub Industry: The specific sub-industry group assigned to a company by the Global Industry Classification Standard (GICS) that best defines its business operations
# - Current Price: Current stock price in dollars
# - Price Change: Percentage change in the stock price in 13 weeks
# - Volatility: Standard deviation of the stock price over the past 13 weeks
# - ROE: A measure of financial performance calculated by dividing net income by shareholders' equity (shareholders' equity is equal to a company's assets minus its debt)
# - Cash Ratio: The ratio of a  company's total reserves of cash and cash equivalents to its total current liabilities
# - Net Cash Flow: The difference between a company's cash inflows and outflows (in dollars)
# - Net Income: Revenues minus expenses, interest, and taxes (in dollars)
# - Earnings Per Share: Company's net profit divided by the number of common shares it has outstanding (in dollars)
# - Estimated Shares Outstanding: Company's stock currently held by all its shareholders
# - P/E Ratio: Ratio of the company's current stock price to the earnings per share 
# - P/B Ratio: Ratio of the company's stock price per share by its book value per share (book value of a company is the net difference between that company's total assets and total liabilities)
# 
# ### Importing necessary libraries and data

# In[1]:


# suppress all warnings
import warnings
warnings.filterwarnings("ignore")

#import libraries needed for data manipulation
import pandas as pd
import numpy as np

pd.set_option('display.float_format', lambda x: '%.3f' % x)

#import libraries needed for data visualization

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# unlimited number of displayed columns, limit of 200 for displayed rows
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 200)


# to scale the data using z-score
from sklearn.preprocessing import StandardScaler

# to compute distances
from scipy.spatial.distance import cdist, pdist

# to perform k-means clustering and compute silhouette scores
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# to visualize the elbow curve and silhouette scores
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

# to perform hierarchical clustering, compute cophenetic correlation, and create dendrograms
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet


# ## Data Overview
# 
# - Observations
# - Sanity checks

# In[2]:


#import dataset named 'stock_data.csv'

stock = pd.read_csv('stock_data.csv')

# read first five rows of the dataset

stock.head()


# In[3]:


stock.shape


# In[4]:


stock.info()


# In[5]:


stock.sample(n=10, random_state=1)


# In[6]:


stock.describe().T


# In[7]:


stock.isnull().sum()


# In[8]:


stock.duplicated().sum()


# In[9]:


# create a copy of the data so that the original dataset is not changed.

df = stock.copy()


# **Observations**
# 
# - 0 null or duplicate values in the dataset.
# - Ticker Symbol identifies stock individual is investing in.
# - GICS (Global Industry Classification Standard) Sector & Sub Industry, as well as Security and Ticker Symbol are object type. Remaining variables are numeric.
# - Net Income averages at ~ 1.5 billion, Net Cash Flow averages at  ~ 55 million.
# 
# ## Exploratory Data Analysis (EDA)
# 
# - EDA is an important part of any project involving data.
# - It is important to investigate and understand the data better before building a model with it.
# 
# **Leading Questions**: _Done within Bivariate analysis section_
# 1. What does the distribution of stock prices look like?
# 2. The stocks of which economic sector have seen the maximum price increase on average?
# 3. How are the different variables correlated with each other?
# 4. Cash ratio provides a measure of a company's ability to cover its short-term obligations using only cash and cash equivalents. How does the average cash ratio vary across economic sectors?
# 5. P/E ratios can help determine the relative value of a company's shares as they signify the amount of money an investor is willing to invest in a single share of a company per dollar of its earnings. How does the P/E ratio vary, on average, across economic sectors?
# 
# ### Univariate Analysis

# In[10]:


# define a function to plot a boxplot and a histogram along the same scale

def histbox(data, feature, figsize=(12, 7), kde=False, bins=None):
    """
    Boxplot and histogram combined
    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (12,7))
    kde: whether to show the density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (box, hist) = plt.subplots(
        nrows=2,                                            # Number of rows of the subplot grid = 2
                                                                # boxplot first then histogram created below 
        sharex=True,                                        # x-axis same among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},        # boxplot 1/3 height of histogram
        figsize=figsize,                                    # figsize defined above as (12, 7)
    )  
    # defining boxplot inside function, so when using it say histbox(df, 'cost'), df: data and cost: feature
    
    sns.boxplot(
        data=data, x=feature, ax=box, showmeans=True, color="chocolate"
    )  # showmeans makes mean val on boxplot have star, ax = 
    sns.histplot(
        data=data, x=feature, kde=kde, ax=hist, bins=bins, color = "darkgreen"
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=hist, color = "darkgreen"
    )  # For histogram if there are bins in potential graph 
    
    # add vertical line in histogram for mean and median
    hist.axvline( 
        data[feature].mean(), color="purple", linestyle="--"
    )  # Add mean to the histogram
    hist.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram


# In[11]:


# define a function to create labeled barplots

def bar(data, feature, perc=False, n=None):
    """
    Barplot with percentage at the top

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n category levels (default is None, i.e., display all levels)
    """

    total = len(data[feature])  # length of the column
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 1, 5))
    else:
        plt.figure(figsize=(n + 1, 5))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n].sort_values(),
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )  # percentage of each class of the category
        else:
            label = p.get_height()  # count of each level of the category

        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )  # annotate the percentage
plt.show()  # show the plot


# In[12]:


histbox(df, 'Current Price')


# In[13]:


histbox(df, 'Price Change')


# In[14]:


histbox(df, 'Volatility')


# In[15]:


histbox(df, 'ROE')


# In[16]:


histbox(df, 'Cash Ratio')


# In[17]:


histbox(df, 'Net Cash Flow')


# In[18]:


histbox(df, 'Net Income')


# In[19]:


histbox(df, 'Earnings Per Share')


# In[20]:


histbox(df, 'Estimated Shares Outstanding')


# In[21]:


histbox(df, 'P/E Ratio')


# In[22]:


histbox(df, 'P/B Ratio')


# In[23]:


bar(df,'GICS Sector', perc=True)


# In[24]:


bar(df, 'GICS Sub Industry', perc=True, n=15)


# **Observations**
# 
# - Current Price: right skewed distribution, average around 80 
# - Price Change: near normal distribution, average around 4
# - Volatility: right skew, high mode around 1.0, mean 1.5
# - ROE: heavy right skew, average around 40
# - Cash Ratio: right skew, average around 70
# - Net Cash Flow/Net Income: similar normal distributions
# - Earnings Per Share: near normal distribution, average 2.77
# - Estimated Shares Outstanding: right skew, average around 577 million
# - P/E Ratio: average at 32.6
# - P/B Ratio: average at -1.72
# 
# - GICS most common sector is Industrials
# - GICS most common sub-industry is Oil & Gas Exploration & Production
# 
# ### Bivariate Analysis
# 
# #### **Leading Question 3: How are the different variables correlated with each other?**
# 

# In[25]:


# correlation check
plt.figure(figsize=(15, 7))
sns.heatmap(
    df.corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral"
)
plt.show()


# - Medium/high correlations:
#     - Current Price - Earnings Per Share
#     - Net Income - Estimated Shares Outstanding
#     - Net Income - Earnings Per Share

# #### **Leading Question 1: What does the distribution of stock prices look like?**

# In[26]:


df['Current Price'].describe()


# In[27]:


histbox(df, 'Current Price')


# **Observations**
# 
# - Majority of prices fall between 38 and 92 dollars.
# - Many outliers, highest being over 1200 dollars.
# 
# #### **Leading Question 2: The stocks of which economic sector have seen the maximum price increase on average?**

# In[28]:


plt.figure(figsize=(15,8))
sns.barplot(data=df, x='GICS Sector', y='Price Change', ci=False)
plt.xticks(rotation=90)
plt.show()


# In[29]:


df.groupby("GICS Sector")["Price Change"].describe()


# In[30]:


# Check volatility (standard deviation of stock price) to better explain certain sector price changes behavior 

df.groupby("GICS Sector")["Volatility"].describe()


# **Observations**
# 
# - Highest postive percentage change was in the Health Care sector, well over 9.5 in the 13 week period.
# - Only negative percentage change was in Energy sector, with the highest volatility (standard dev).
# - Lowest positive percentage change was in Utilities sector.
# 
# #### **Leading Question 4: Cash ratio provides a measure of a company's ability to cover its short-term obligations using only cash and cash equivalents. How does the average cash ratio vary across economic sectors?** 

# In[31]:


plt.figure(figsize=(15,8))
sns.barplot(data=df, x='GICS Sector', y='Cash Ratio', ci=False)
plt.xticks(rotation=90)
plt.show()


# In[32]:


df.groupby("GICS Sector")["Cash Ratio"].describe()


# **Observations**
# 
# - Cash Ratio: Company's total reserves of cash and cash equivalents : total current liabilities
# - Highest is in IT Sector, lowest is in Utilities Sector
# 
# #### **Leading Question 5: P/E ratios can help determine the relative value of a company's shares as they signify the amount of money an investor is willing to invest in a single share of a company per dollar of its earnings. How does the P/E ratio vary, on average, across economic sectors?**

# In[33]:


plt.figure(figsize=(15,8))
sns.barplot(data=df, x='GICS Sector', y='P/E Ratio', ci=False)
plt.xticks(rotation=90)
plt.show()


# **Observations**
# 
# - P/E Ratio: Stock price : Earnings Per Share
# - Highest by a large margin is in Energy sector, lowest is Telecommunications Services
# 
# ## Data Preprocessing
# 
# - Missing value treatment (not needed, no missing values)
# - Feature engineering (if needed)
# - Outlier detection and treatment
# - Preparing data for modeling
# - Any other preprocessing steps (if needed)
# 
# ### Outlier Detection and treatment

# In[34]:


# outlier detection using boxplot

num_cols = df.select_dtypes(include=np.number).columns.tolist()
plt.figure(figsize=(10, 8))

for i, variable in enumerate(num_cols):
    plt.subplot(4, 4, i + 1)
    plt.boxplot(df[variable], whis=1.5)
    plt.tight_layout()
    plt.title(variable)

plt.show()


# **Observations:**
# 
# - There are quite a few outliers in the data, notably in net income and estimated shares outstanding.
# - However, they are proper values and reflect the market. We will scale the data before proceeding with clustering.

# In[35]:


# Scaling the data set before clustering

scaler = StandardScaler()
subset = df[num_cols].copy()
subset_scaled = scaler.fit_transform(subset)


# In[36]:


subset_scaled_df = pd.DataFrame(subset_scaled, columns=subset.columns)


# ## K-means Clustering

# In[37]:


k_means = subset_scaled_df.copy()


# In[38]:


clusters = range(1, 15)
meanDistortions = []

for k in clusters:
    model = KMeans(n_clusters=k, random_state=1)
    model.fit(subset_scaled_df)
    prediction = model.predict(k_means)
    distortion = (
        sum(np.min(cdist(k_means, model.cluster_centers_, "euclidean"), axis=1))
        / k_means.shape[0]
    )

    meanDistortions.append(distortion)

    print("Number of Clusters:", k, "\tAverage Distortion:", distortion)

plt.plot(clusters, meanDistortions, "bx-")
plt.xlabel("k")
plt.ylabel("Average Distortion")
plt.title("Selecting k with the Elbow Method", fontsize=20)
plt.show()


# In[39]:


model = KMeans(random_state=1)
visualizer = KElbowVisualizer(model, k=(1, 15), timings=True)
visualizer.fit(k_means) 
visualizer.show() 


# **The appropriate value of k from the elbow curve is 6. Let's check the silhouette scores.**

# In[40]:


sil_score = []
cluster_list = range(2, 15)
for n_clusters in cluster_list:
    clusterer = KMeans(n_clusters=n_clusters, random_state=1)
    preds = clusterer.fit_predict((subset_scaled_df))
    score = silhouette_score(k_means, preds)
    sil_score.append(score)
    print("For n_clusters = {}, the silhouette score is {})".format(n_clusters, score))

plt.plot(cluster_list, sil_score)
plt.show()


# In[41]:


model = KMeans(random_state=1)
visualizer = KElbowVisualizer(model, k=(2, 15), metric="silhouette", timings=True)
visualizer.fit(k_means)  
visualizer.show()


# In[42]:


# finding optimal no. of clusters with silhouette coefficients
visualizer = SilhouetteVisualizer(KMeans(2, random_state=1))
visualizer.fit(k_means)
visualizer.show()


# In[43]:


# finding optimal no. of clusters with silhouette coefficients
visualizer = SilhouetteVisualizer(KMeans(3, random_state=1))
visualizer.fit(k_means)
visualizer.show()


# In[44]:


# finding optimal no. of clusters with silhouette coefficients
visualizer = SilhouetteVisualizer(KMeans(4, random_state=1))
visualizer.fit(k_means)
visualizer.show()


# **Let's take 3 as the appropriate no. of clusters as the silhouette score is high enough.**

# In[45]:


final = KMeans(n_clusters=3, random_state=0)
final.fit(k_means)


# In[46]:


# creating a copy of the original data
df1 = df.copy()

# adding final k-means model cluster labels 
k_means["KM_segments"] = final.labels_
df1["KM_segments"] = final.labels_


# ### Cluster Profiles

# In[47]:


cluster_profile = df1.groupby("KM_segments").mean()


# In[48]:


cluster_profile["count_in_each_segment"] = (
    df1.groupby("KM_segments")["Security"].count().values  ## Complete the code to groupby the cluster labels
)


# In[49]:


cluster_profile.style.highlight_max(color="lightgreen", axis=0)


# In[50]:


# let's see the names of the companies in each cluster
for cl in df1["KM_segments"].unique():
    print("In cluster {}, the following companies are present:".format(cl))
    print(df1[df1["KM_segments"] == cl]["Security"].unique())
    print()


# In[51]:


df1.groupby(["KM_segments", "GICS Sector"])['Security'].count()


# In[52]:


fig, axes = plt.subplots(1, 5, figsize=(16, 6))
fig.suptitle("Boxplot of numerical variables for each cluster")
counter = 0
for ii in range(5):
    sns.boxplot(ax=axes[ii], y=df1[num_cols[counter]], x=df1["KM_segments"])
    counter = counter + 1

fig.tight_layout(pad=2.0)


# In[53]:


df1.groupby("KM_segments").mean().plot.bar(figsize=(15, 6))


# ### Insights
# 
# - Cluster 0:
#     - Largest sectors are: Industrials, Financials, Consumer Discretionary
#     - Largest number of companies (notable: several banks, petrol, holdings)
#     - Current price has most outliers
#     - Moderate price change
#     - Low votality
#     - Many outliers in ROE and Cash Ratio
# - Cluster 1:
#     - Largest sectors are: Financials, Health Care
#     - Notable companies: Merck, Pfizer, Exxon, Verizon, Wells Fargo, Facebook
#     - Current price is highest
#     - Low price change
#     - Low votality 
#     - Moderate ROE and Cash Ratio
# - Cluster 2:
#     - Largest sectors are: Energy, Information Technology 
#     - Notable companies: Amazon, Netflix, several oil corporations
#     - Current price is lowest
#     - Low price change
#     - Very high votality
#     - High ROE, Moderate Cash Ratio
#     
# # Hierarchical Clustering

# In[54]:


hc_df = subset_scaled_df.copy()


# In[55]:


# list of distance metrics
distance_metrics = ["euclidean", "chebyshev", "mahalanobis", "cityblock"]

# list of linkage methods
linkage_methods = ["single", "complete", "average", "weighted"]

high_cophenet_corr = 0
high_dm_lm = [0, 0]

for dm in distance_metrics:
    for lm in linkage_methods:
        Z = linkage(hc_df, metric=dm, method=lm)
        c, coph_dists = cophenet(Z, pdist(hc_df))
        print(
            "Cophenetic correlation for {} distance and {} linkage is {}.".format(
                dm.capitalize(), lm, c
            )
        )
        if high_cophenet_corr < c:
            high_cophenet_corr = c
            high_dm_lm[0] = dm
            high_dm_lm[1] = lm


# In[56]:


# printing the combination of distance metric and linkage method with the highest cophenetic correlation
print(
    "Highest cophenetic correlation is {}, which is obtained with {} distance and {} linkage.".format(
        high_cophenet_corr, high_dm_lm[0].capitalize(), high_dm_lm[1]
    )
)


# **Let's explore different linkage methods with Euclidean distance only.**

# In[57]:


# list of linkage methods
linkage_methods = ["single", "complete", "average", "centroid", "ward", "weighted"]

high_cophenet_corr = 0
high_dm_lm = [0, 0]

for lm in linkage_methods:
    Z = linkage(hc_df, metric="euclidean", method=lm)
    c, coph_dists = cophenet(Z, pdist(hc_df))
    print("Cophenetic correlation for {} linkage is {}.".format(lm, c))
    if high_cophenet_corr < c:
        high_cophenet_corr = c
        high_dm_lm[0] = "euclidean"
        high_dm_lm[1] = lm


# In[58]:


# printing the combination of distance metric and linkage method with the highest cophenetic correlation
print(
    "Highest cophenetic correlation is {}, which is obtained with {} linkage.".format(
        high_cophenet_corr, high_dm_lm[1]
    )
)


# **Let's view the dendrograms for the different linkage methods with Euclidean distance.**

# In[59]:


# list of linkage methods
linkage_methods = ["single", "complete", "average", "centroid", "ward", "weighted"]

# lists to save results of cophenetic correlation calculation
compare_cols = ["Linkage", "Cophenetic Coefficient"]

# to create a subplot image
fig, axs = plt.subplots(len(linkage_methods), 1, figsize=(15, 30))

# We will enumerate through the list of linkage methods above
# For each linkage method, we will plot the dendrogram and calculate the cophenetic correlation
for i, method in enumerate(linkage_methods):
    Z = linkage(hc_df, metric="euclidean", method=method)

    dendrogram(Z, ax=axs[i])
    axs[i].set_title(f"Dendrogram ({method.capitalize()} Linkage)")

    coph_corr, coph_dist = cophenet(Z, pdist(hc_df))
    axs[i].annotate(
        f"Cophenetic\nCorrelation\n{coph_corr:0.2f}",
        (0.80, 0.80),
        xycoords="axes fraction",
    )


# **Observations**
# 
# - The cophenetic correlation is highest for average and centroid linkage methods, followeed by single and weighted.
# - We will move ahead with average linkage.
# - 6 appears to be the appropriate number of clusters from the dendrogram for average linkage. 

# In[60]:


HCmodel = AgglomerativeClustering(n_clusters=6, affinity="euclidean", linkage="average")
HCmodel.fit(hc_df)


# In[61]:


# creating a copy of the original data
df2 = df.copy()

# adding hierarchical cluster labels to the original and scaled dataframes
hc_df["HC_segments"] = HCmodel.labels_
df2["HC_segments"] = HCmodel.labels_


# ### Cluster Profiles

# In[62]:


cluster_profile2 = df2.groupby("HC_segments").mean()


# In[63]:


cluster_profile2["count_in_each_segment"] = (
    df2.groupby("HC_segments")["Security"].count().values
)


# In[64]:


cluster_profile2.style.highlight_max(color="lightgreen", axis=0)


# In[65]:


# let's see the names of the companies in each cluster
for cl in df2["HC_segments"].unique():
    print("In cluster {}, the following companies are present:".format(cl))
    print(df2[df2["HC_segments"] == cl]["Security"].unique())
    print()


# In[66]:


df2.groupby(["HC_segments", "GICS Sector"])['Security'].count()


# In[67]:


fig, axes = plt.subplots(3, 4, figsize=(20, 20))
counter = 0

for ii in range(3):
    for jj in range(4):
        if counter < 11:
            sns.boxplot(
                ax=axes[ii][jj],
                data=df2,
                y=df2.columns[4+counter],
                x="HC_segments",
            )
            counter = counter + 1

fig.tight_layout(pad=3.0)


# ### Insights
# 
# - Looking at Clusters 0-2 since the rest are very small.
# - Cluster 0 
#     - Largest number of companies by a large margin
#     - Moderate current price
#     - Moderate price change, volatility, ROE, Estimated shares outstanding
#     - Many outliers in Net Cash Flow & Net Income
#     - P/E Ratio on the lower end
# - Cluster 1
#     - Bank of America and Intel Corp.
#     - Low current price
#     - Moderate price change, volatility, ROE, net cash flow
# - Cluster 2
#     - Apache Corp and Chesapeake Energy
#     - Very atypical compared to other clusters
#     - High volatility, ROE
#     - Low earnings per share, estimated shares outstanding, net cash flow, net income
# 
# ## K-means vs Hierarchical Clustering
# 
# - K Means executed immediately, compared to Hierarchial Clustering taking longer
# - Appropriate number of clusters determined to be: 
#     - K Means: 3
#     - Hierarchial: 6 (noted that the latter 3 of the 6 clusters have very little data)
# - More distinct clusters obtained from K Means
# - Cluster 0 in both styles, and Cluster 2 in both styles were very similar
#     - Cluster 0: majority Consumer Staples and Consumer Discretionary
#     - Cluster 2: majority Energy, most atypical when compared to other clusters (high volatility, low price change and net cash flow/income)
#     
# ## Actionable Insights and Recommendations
# 
# - Cluster 0 (K Means and Hierarchial): These are composed of companies that most consumers will encounter in their day to day lives. **Those looking to maximize earnings with little risk should approach this cluster to invest.**
# - Cluster 1 (K Means): Composed of financial groups and health care. During a period of cyclical unemployment, avoid investing in banks and loan holdings. **On the upswing of market expansion, consumers can invest in these areas. With healthcare, look towards market trends and the latest innovations within these companies (notably Merck and Pfizer) to decide when to invest to maximize earnings.** 
# - Cluster 1 (Hierarchial): Composed of Bank of America and Intel. **Invest in periods when consumers can feasibly make a profit.**
# - Cluster 2( K Means and Hierarchial): Composed of majority energy and oil corporations. With high volatility, and a high cash ratio (total cash reserves:total liabilities), **consumers that are more risk seeking may invest during times of a stable economy to maximize chances of making a profit.**
