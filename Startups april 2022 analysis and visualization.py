# %% [markdown]
# ---
# # <b> Data cleaning, Analysing and Visualization using </b>
#
#
#     - Pandas
#     - Matplotlib
#     - seaborn.
#
# <img src = "https://www.bigscal.com/wp-content/uploads/2022/03/data-analysis-and-visualization.jpg" width = "570" height = "250">
#
# <br>
# <br>
#
# >>> Startups may be small companies but they can play a significant role in economic growth.Startup funding, or startup capital, is money that an entrepreneur uses to launch a new business. This money can be used for hiring employees, renting space, buying inventory or other operating expenses that help a business get started.
#
# >>> Here is the report of Startup funding for April 2022.
#
# ***
# ---

# %% [markdown]
# ### <b> Importing necessary `libraries` </b>

# %%
# importing libraries
from thefuzz import process, fuzz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# ---
# ### <b> Import and reading the required Dataset </b>

# %%
# Reading csv file
df = pd.read_csv(
    "Indian Startups - Funding  Investors Data April 2022.csv", encoding='cp1252')

# %%
df.head(10)

# %%
df.tail(10)

# %% [markdown]
# This shows the first and last `10 rows` of the dataset. From these we can easily understand the nature of the dataset.
#
# ---

# %% [markdown]
# ### <b> Getting more info about the dataset. </b>
#
# - Getting shape of the dataset

# %%
df.shape

# %% [markdown]
# >>>> so this dataset consist of 95 rows and 9 columns.

# %% [markdown]
# - Checking the Datatype of each variable in Dataset.

# %%
df.info()

# %% [markdown]
# >>>> Here most of the variables are in "Object" type.

# %% [markdown]
# - Getting percentage of empty cells.

# %%
total_cells = np.product(df.shape)
empty_cells = df.isnull().sum().sum()
percent_empty = (empty_cells/total_cells)*100
percent_empty

# %% [markdown]
# >>>> Nearly `3%` of the data is filled with empty cells.
#

# %% [markdown]
# ---
# # <b> Data Cleaning </b>
# - Checking for duplicates

# %%
df.duplicated().sum()
# the dataset doesn't contain any duplicated cells

# %% [markdown]
#  >>>> The dataset doesn't contain any duplicated cells. So there is no need to drop duplicates.
#
#  ---

# %% [markdown]
# - Handling misplaced cells.

# %%
misplaced_cells = df['Amount'].isnull()
df[misplaced_cells]

# %% [markdown]
# >>>> we can notice that some values are misplaced. Here the `Amount` variable will be useful in analysis. Instead of removing the cells, Replacing them with the corresponding values from respective rows for better analysis.

# %%
df.loc[85, 'Amount'] = '$270,000,000'
df.loc[91, 'Amount'] = '$66,000,000'

# %%
df[misplaced_cells]

# %% [markdown]
# >>>> - In order to analyze `Amount` variable, we need need to convert them to `int` rather than `object`.
# >>>> - So removing the special characters such as  `$`, `,`, `.` and variables such as `Undisclosed` is necessery to convert them as `int`.

# %%
# Removing the special characters ('$',',','.')
df['Amount'] = df['Amount'].str.replace(r'\W', '', regex=True)

# Replacing 'Undisclosed' with '0'
df['Amount'] = df['Amount'].str.replace('Undisclosed', '0')

# Changing datatype
df["Amount"] = df["Amount"].astype('int64')

# %%
df.info()

# %% [markdown]
# >>>> Dtype of `Amount` becomes `int64`

# %%
df.isna().sum()

# %% [markdown]
# >>>> Now let's fill the empty cells with `Undisclosed`

# %%
df.fillna('Undisclosed', inplace=True)
df

# %%
df.isnull().sum()

# %% [markdown]
# - Handling inconsistent data.
#

# %% [markdown]
# >>>> Looks like `Bangalore`, `Banglore`, `Bengaluru` denotes the same location `Banglore`.

# %%

# %%
# Getting unique values.
Locations = df['Location'].unique()
Locations.sort()
Locations


# %%
# Getting matches for 'Banglore'
matches = process.extract('Banglore', Locations,
                          limit=5, scorer=fuzz.token_sort_ratio)
matches

# %%
# Getting almost perfect matches for "Banglore"
perfect_matches = [matches[0] for matches in matches if matches[1] >= 58]
perfect_matches

# %%
# Getting the position of rows from our dataFrame.
rows_with_perfect_matches = df['Location'].isin(perfect_matches)
rows_with_perfect_matches

# %%
# Replacing
df.loc[rows_with_perfect_matches, 'Location'] = 'Banglore'

# %%
locations = df['Location'].unique()
locations.sort()
locations

# %% [markdown]
# >>>> Now `Bangalore`, `Banglore`, `Bengaluru` is replaced with `Banglore`.
#
# >>>> Now the dataset is cleaned and ready for analysis.
#
# ---

# %% [markdown]
# # <b> Data analysis and Visualization </b>
#
# >>>> For convenience I'm adding a extra variable to the dataset - `Amont_million`

# %%
# Getting amount in million.
df['Amount_million'] = (df['Amount']/1000000)
df['Amount_million'].head()

# %%
df.head()

# %% [markdown]
# ### FUNDING PER REGION

# %%
funding_per_region = df.groupby(
    ['Location'])['Amount_million'].sum().reset_index()
funding_per_region = funding_per_region.sort_values(
    'Amount_million', ascending=False).head(10)
funding_per_region

# %%
fig, ax = plt.subplots(figsize=(10, 5))
sns.set_palette('Set2')
sns.set_style('dark')
plt.title("FUNDING PER REGION", family='serif', color='k', size='large')
sns.barplot(x='Amount_million', y='Location', data=funding_per_region)
plt.show()

# %% [markdown]
# This barplot shows the top 10 fundings per region.
# - Startups from `Palo Alto` a city of California received fundings over $18,000 million
# - Followed by stratups founded at `Banglore` received fundings over $9,000 million.
# ---

# %% [markdown]
# ### FUNDING OVER SECTORS.

# %%
fund_sector = df.groupby('Sector')['Amount_million'].sum().reset_index()
fund_sector = fund_sector.sort_values('Amount_million', ascending=False).head()
fund_sector

# %%
fig, ax = plt.subplots(figsize=(10, 5))
sns.set_palette('Set2')
sns.set_style('dark')
plt.title("TOP 5 FUNDING OVER SECTOR", family='serif', color='k', size='large')
sns.barplot(y='Amount_million', x='Sector', data=fund_sector.head())
plt.show()

# %% [markdown]
# This barplot shows the fundings over different sectors.
# - Startups based on `Financial services` got the most funding followed by `Travel arrangements` and `Logistics Sector`
#
# ---

# %% [markdown]
# ### FUNDING OVER COMPANY.

# %%
company = df.groupby('Company Name')['Amount_million'].sum().reset_index()
company = company.sort_values('Amount_million', ascending=False)
company.head()

# %%
fig, ax = plt.subplots(figsize=(10, 5))
sns.set_palette('Set2')
sns.set_style('dark')
plt.title("FUNDING OVER COMPANIES", family='serif', color='k', size='large')
sns.barplot(x='Company Name', y='Amount_million', data=company.head())
plt.show()

# %% [markdown]
# This barplot shows top 5 fundings over Companies.
# - `Accel` got funding over $18,000 million.
# - Followed by `Ola` - $5,000 million and `Mahindra logistics` - $2,500 million.
# ---

# %% [markdown]
# ### NUMBER OF STARTUPS PER REGION

# %%
loc_order = df['Location'].value_counts().head(10).index
fig, ax = plt.subplots(figsize=(10, 5))
sns.set_palette('Set2')
sns.set_style('dark')
plt.title("STARTUPS PER REGION", family='serif', color='k', size='large')
sns.countplot(x='Location', data=df, order=loc_order)
plt.show()


# %% [markdown]
# - Nearly 25 companies from `Banglore` raised funds.
# - Followed by `Mumbai` - nearly 10 and `Gurgaon` - nearly 8 companies.
#
# ---

# %% [markdown]
# ### FUNDINGS PER SECTOR

# %%
stage_order = df['Stage'].value_counts().head().index
fig, ax = plt.subplots(figsize=(10, 5))
sns.set_style('dark')
sns.countplot(x='Stage', data=df, order=stage_order, palette='Set3')
plt.title("FUNDINGS PER SECTOR", family='serif', color='k', size='large')
plt.show()


# %% [markdown]
# Here is the top 5 Fundings over Sectors.
# - Here most of the sectors are left disclosed.
# - Leaving that, there are nearly 10 `Series F` and nearly 10 `Series E` sector.
# ---

# %% [markdown]
# ### SATRTUPS OVER TIME

# %%
fig, ax = plt.subplots(figsize=(10, 5))
sns.set_style('dark')
sns.histplot(x='Founded', data=df, color='teal')
plt.title("STARTUPS OVER TIME", family='serif', color='k', size='large')
plt.show()

# %% [markdown]
# - Large number of startups that founded after year `2000` raised funds while comparing to the startups that founded before year 2000.
# ---
# ---
