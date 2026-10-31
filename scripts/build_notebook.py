"""Builds notebooks/Walmart_CLT_Case_Study.ipynb from the cell list below."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

# ---------------------------------------------------------------- 0. Title
md(r"""
# Walmart Black Friday: Do Women Spend More Than Men?
### Confidence Intervals and the Central Limit Theorem applied to 550,068 transactions

**Author:** Guntuku Hariprasad &nbsp;|&nbsp; **Program:** Scaler, Business Case Study &nbsp;|&nbsp; **Tools:** Python, pandas, NumPy, SciPy, Matplotlib, Seaborn

---

## Executive summary

| Question | Answer (95% confidence, per transaction) |
|---|---|
| Do women spend more per transaction than men? | **No.** Men average **9,437.5** vs women **8,734.6**, a gap of about **703 (8%)**. |
| Do the male and female confidence intervals overlap? | **No.** Female [8,709, 8,760] and Male [9,422, 9,453] are far apart at 90%, 95% and 99%. |
| Does marital status change spend per transaction? | **No.** Unmarried 9,265.9 vs Married 9,261.2; the intervals overlap almost completely. |
| Does age change spend per transaction? | **Yes, modestly.** Spend rises with age: 0-17 (8,934) < 18-25 (9,170) < 26-35 (9,253) ≈ 36-50 (9,295) < 51+ (9,464). |
| What drives the gender gap? | **Basket mix, not price sensitivity.** Men buy more of high-ticket Category 1; women buy more of lower-ticket Categories 5 and 8. Within the same product category, women spend just as much as men; the entire gap is explained by product mix. |
| Where is the revenue? | Men are 75% of transactions and 77% of Black Friday revenue; customers aged 26-35 are the single most valuable group. |

**Bottom line for Walmart:** men and women behave differently in *what* they buy, not just *how much*. The biggest lever is to move women into higher-value categories and to win more female shoppers, while keeping the 26-35 core and the high-ticket-friendly 51+ segment engaged.
""")

md(r"""
## Table of contents
1. Problem statement and approach
2. Data loading, structure and basic metrics
3. Non-graphical analysis: value counts and unique values
4. Missing values and outlier detection
5. Visual analysis: univariate, bivariate and correlation
6. Gender: are women spending more per transaction than men?
7. Gender: Central Limit Theorem and confidence intervals (90 / 95 / 99%)
8. Marital status: Married vs Unmarried
9. Age: life-stage bins (0-17, 18-25, 26-35, 36-50, 51+)
10. Bonus: the customer-level view (total spend per shopper)
11. Final insights
12. Recommendations for Walmart
""")

# ---------------------------------------------------------------- 1. Problem
md(r"""
## 1. Problem statement and approach

**Business problem.** Walmart's management wants to understand how purchase amount varies with customer gender and other attributes, and specifically: *do women spend more on Black Friday than men?* We assume the population is **50 million male and 50 million female customers**, and the Black Friday transactions in this dataset are a sample from that population.

**Why statistics is needed.** The data is a sample, so any average we compute is an *estimate*. A different sample would give a slightly different number. A **confidence interval (CI)** gives a range that, with a chosen level of confidence (90%, 95% or 99%), contains the true population average. The **Central Limit Theorem (CLT)** tells us that the average of a reasonably large sample is approximately normally distributed, even when individual purchases are not, which is what lets us build those intervals.

**Approach.**
1. Profile the data: shape, types, summary statistics, missing values, outliers.
2. Explore it visually to understand distributions and relationships.
3. For each segment (Gender, Marital Status, Age): bootstrap sample means at several sample sizes to watch the CLT at work, then compute 90/95/99% CIs and check whether the intervals overlap.
4. Translate the results into plain-language recommendations.

**Unit of analysis.** The case asks about spend **per transaction**, so the main analysis uses each row's `Purchase`. Section 10 repeats the key test on **total spend per customer** as a robustness check.
""")

# ---------------------------------------------------------------- 2. Load
md("## 2. Data loading, structure and basic metrics")
code(r"""
import warnings, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
pd.set_option("display.width", 140)

# Consistent, colour-blind-checked palette
BLUE, ORANGE, AQUA, VIOLET, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#8a8984"
GENDER_PAL  = {"Female": ORANGE, "Male": BLUE}
MARITAL_PAL = {"Unmarried": AQUA, "Married": VIOLET}
AGE_ORDER   = ["0-17", "18-25", "26-35", "36-50", "51+"]
AGE_PAL     = dict(zip(AGE_ORDER, ["#b7d3f2", "#7fb0e8", "#2a78d6", "#1d58a3", "#103a70"]))

sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
plt.rcParams.update({"figure.dpi": 90, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 11, "grid.color": "#e6e5e0"})

IMG_DIR = "../images"
os.makedirs(IMG_DIR, exist_ok=True)
def save(fig, name):
    fig.savefig(f"{IMG_DIR}/{name}.png", dpi=130, bbox_inches="tight")

RNG = np.random.default_rng(42)   # reproducible sampling
""")
code(r"""
df = pd.read_csv("../data/walmart_data.csv")
print("Shape:", df.shape)
df.head()
""")
code(r"""
df.info()
""")
md(r"""
**Observations on structure**
- **550,068 rows × 10 columns.** Each row is one product purchased by one customer on Black Friday.
- `User_ID`, `Occupation`, `Marital_Status` and `Product_Category` are stored as integers, but they are **codes / labels**, not quantities. Averaging them is meaningless, so they are converted to `category`.
- `Purchase` is the only true numeric (continuous) variable.
- Converting the 9 descriptive columns to `category` cuts memory sharply and makes intent explicit.
""")
code(r"""
mem_before = df.memory_usage(deep=True).sum() / 1024**2
cat_cols = ["User_ID", "Product_ID", "Gender", "Age", "Occupation", "City_Category",
            "Stay_In_Current_City_Years", "Marital_Status", "Product_Category"]
df[cat_cols] = df[cat_cols].astype("category")

# Readable labels for reporting
df["Gender"] = df["Gender"].cat.rename_categories({"F": "Female", "M": "Male"})
df["Marital_Status"] = df["Marital_Status"].cat.rename_categories({0: "Unmarried", 1: "Married"})

# Life-stage age bins requested in the case (original bins merged)
life_stage = {"0-17": "0-17", "18-25": "18-25", "26-35": "26-35", "36-45": "36-50",
              "46-50": "36-50", "51-55": "51+", "55+": "51+"}
df["Life_Stage"] = pd.Categorical(df["Age"].map(life_stage), categories=AGE_ORDER, ordered=True)
df["Age"] = df["Age"].cat.reorder_categories(["0-17","18-25","26-35","36-45","46-50","51-55","55+"], ordered=True)

mem_after = df.memory_usage(deep=True).sum() / 1024**2
print(f"Memory: {mem_before:,.1f} MB -> {mem_after:,.1f} MB")
df.dtypes
""")
code(r"""
df.describe(include="all").T
""")
code(r"""
p = df["Purchase"]
summary = pd.Series({"mean": p.mean(), "median": p.median(), "mean - median": p.mean() - p.median(),
                     "std": p.std(), "min": p.min(), "max": p.max(),
                     "skewness": p.skew(), "kurtosis": p.kurt()})
summary.to_frame("Purchase")
""")
md(r"""
**Statistical summary: observations**
- **Purchase ranges from 12 to 23,961**; the average transaction is **9,264** and the median is **8,047**.
- **Mean > median by ~1,217**, and skewness is **+0.60**, so the distribution is **right-skewed**: a minority of large purchases pulls the average up.
- **5,891 unique customers** bought **3,631 unique products**: on average each customer made ~93 transactions, so the data is rich enough for stable estimates.
- Most frequent values: Male, age 26-35, City B, 1 year in current city, Unmarried, Product Category 5.
""")

# ---------------------------------------------------------------- 3. Non-graphical
md("## 3. Non-graphical analysis: value counts and unique values")
code(r"""
df[cat_cols + ["Life_Stage"]].nunique().to_frame("unique values").T
""")
code(r"""
def vc(col):
    t = df[col].value_counts()
    return pd.DataFrame({"count": t, "share %": (t / len(df) * 100).round(2)})

for col in ["Gender", "Life_Stage", "Age", "City_Category", "Stay_In_Current_City_Years", "Marital_Status"]:
    print(f"\n--- {col} ---")
    print(vc(col).sort_index().to_string())
""")
code(r"""
users = (df.groupby("User_ID", observed=True)
           .agg(Gender=("Gender", "first"), Marital_Status=("Marital_Status", "first"),
                Life_Stage=("Life_Stage", "first"), City=("City_Category", "first"),
                Total_Spend=("Purchase", "sum"), Transactions=("Purchase", "size"),
                Avg_Ticket=("Purchase", "mean"))
           .reset_index())
print("Unique customers by gender:")
print(users["Gender"].value_counts().to_string())
print("\nTransactions per customer, by gender:")
users.groupby("Gender", observed=True)[["Transactions", "Total_Spend", "Avg_Ticket"]].mean()
""")
md(r"""
**Observations**
- **Gender:** 75.3% of transactions are by men and 24.7% by women. By people, **4,225 men vs 1,666 women (72% / 28%)**. Men also transact more often (~98 vs ~82 items each).
- **Age:** 26-35 is the largest group (39.9% of transactions); 0-17 is the smallest (2.7%).
- **City:** City B has the most transactions (42%), then C (31%) and A (27%).
- **Stay in city:** 1 year is most common (35%); newcomers (0 years) are the smallest group.
- **Marital status:** 59% unmarried, 41% married.
- **Product category:** 20 categories; 1, 5 and 8 account for ~74% of all transactions.
- The **sample is not 50/50 by gender**, while the stated population is. That is exactly why we estimate *averages* with confidence intervals per group, rather than comparing raw totals.
""")

# ---------------------------------------------------------------- 4. Missing / outliers
md("## 4. Missing values and outlier detection")
code(r"""
missing = df.isnull().sum()
print("Missing values per column:\n", missing.to_string())
print("\nTotal missing:", missing.sum(), "| Duplicate rows:", df.duplicated().sum())
""")
code(r"""
q1, q3 = p.quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outliers = df[(p < lower) | (p > upper)]
print(f"Q1 = {q1:,.0f} | Q3 = {q3:,.0f} | IQR = {iqr:,.0f}")
print(f"Whiskers: [{lower:,.0f}, {upper:,.0f}]")
print(f"Outliers: {len(outliers):,} rows ({len(outliers)/len(df):.2%}), all on the high side (>{upper:,.0f})")
print("\nOutliers by gender (% of that gender's transactions):")
print((outliers["Gender"].value_counts() / df["Gender"].value_counts() * 100).round(2).to_string())
print("\nTop product categories among outliers:")
print(outliers["Product_Category"].value_counts().loc[lambda s: s > 0].to_string())
""")
code(r"""
fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
sns.boxplot(x=p, ax=axes[0], color=BLUE, width=0.4, fliersize=2)
axes[0].set_title("Purchase: overall boxplot")
sns.boxplot(data=df, x="Purchase", y="Gender", palette=GENDER_PAL, ax=axes[1], width=0.5, fliersize=2)
axes[1].set_title("Purchase by gender")
clipped = p.clip(p.quantile(0.05), p.quantile(0.95))
sns.boxplot(x=clipped, ax=axes[2], color=AQUA, width=0.4)
axes[2].set_title("After clipping at 5th / 95th percentile")
plt.tight_layout(); save(fig, "01_outliers_boxplots"); plt.show()

print(f"Mean before clipping: {p.mean():,.1f} | after: {clipped.mean():,.1f} | median: {p.median():,.0f}")
""")
md(r"""
**Missing values and outliers: observations**
- **No missing values and no duplicate rows**, so no imputation is required.
- The IQR rule flags **2,677 transactions (0.49%)** above **21,400**, all on the high side. They come almost entirely from high-ticket product categories (10, 15 and 9; Category 10 alone is 85%), so they are **genuine large purchases, not data errors**.
- Clipping at the 5th/95th percentile moves the mean by only ~7 (9,264 → 9,257). Because the outliers are real and few, and because the CLT works on *sample means* (which are robust here), **we keep all rows** for the confidence-interval analysis. Removing real high spenders would bias the estimate of what customers actually spend.
""")

# ---------------------------------------------------------------- 5. Visual
md("## 5. Visual analysis\n### 5.1 Univariate: the continuous variable (`Purchase`)")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(15, 3.8))
sns.histplot(p, bins=60, kde=True, color=BLUE, ax=axes[0], edgecolor="white", linewidth=0.3)
axes[0].axvline(p.mean(), color="#0b0b0b", ls="--", lw=1.5, label=f"Mean {p.mean():,.0f}")
axes[0].axvline(p.median(), color=ORANGE, ls="-", lw=1.5, label=f"Median {p.median():,.0f}")
axes[0].legend(); axes[0].set_title("Distribution of purchase amount (histogram + KDE)")
sns.kdeplot(data=df, x="Purchase", hue="Gender", palette=GENDER_PAL, common_norm=False, fill=True, alpha=0.25, ax=axes[1])
axes[1].set_title("Purchase distribution by gender (density)")
plt.tight_layout(); save(fig, "02_purchase_distribution"); plt.show()
""")
md(r"""
**Observation:** Purchase is **not normal**. It is right-skewed and **multi-modal**, with peaks near ~6-8K, ~11K, ~15-16K and ~19-20K. These peaks line up with the price bands of different product categories, so the distribution is really a *mixture* of category price levels. The male curve has more mass in the 15K-20K range, the female curve more in the 5K-8K range. Because individual purchases are far from normal, we will rely on the **CLT** (distribution of *sample means*) rather than on the raw distribution.
""")
md("### 5.2 Univariate: categorical variables (count plots)")
code(r"""
cols = ["Gender", "Life_Stage", "Marital_Status", "City_Category", "Stay_In_Current_City_Years", "Occupation"]
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for ax, col in zip(axes.flat, cols):
    order = df[col].cat.categories
    sns.countplot(data=df, x=col, order=order, color=BLUE, ax=ax, edgecolor="white")
    ax.set_title(f"Transactions by {col.replace('_', ' ')}"); ax.set_xlabel(""); ax.set_ylabel("count")
    if col != "Occupation":
        for c in ax.containers:
            ax.bar_label(c, labels=[f"{v/len(df):.0%}" for v in c.datavalues], fontsize=8, padding=2)
    else:
        ax.tick_params(axis="x", labelsize=8)
plt.tight_layout(); save(fig, "03_countplots_categorical"); plt.show()
""")
code(r"""
fig, ax = plt.subplots(figsize=(15, 3.6))
pc = df["Product_Category"].value_counts().sort_index()
ax.bar(pc.index.astype(str), pc.values, color=BLUE, edgecolor="white")
ax2_vals = df.groupby("Product_Category", observed=True)["Purchase"].mean()
for x, (cnt, avg) in enumerate(zip(pc.values, ax2_vals.reindex(pc.index).values)):
    ax.text(x, cnt, f"{avg/1000:.1f}K", ha="center", va="bottom", fontsize=8, color="#52514e")
ax.set_title("Transactions by product category (label = average ticket size)")
ax.set_xlabel("Product category"); ax.set_ylabel("count")
plt.tight_layout(); save(fig, "04_product_category"); plt.show()
""")
md(r"""
**Observations (univariate)**
- **Men dominate transaction volume (75%)**, so total revenue comparisons would be misleading. We must compare *averages*.
- **26-35 is the core shopper (40%)**, followed by 36-50 (28%). Teenagers (0-17) are only 3%.
- **Unmarried shoppers (59%)** outnumber married ones.
- **City B** leads in volume; **1-year residents** are the largest tenure group, suggesting recent movers are active shoppers.
- **Occupations 4, 0 and 7** are the most frequent; 8 is rare.
- **Product categories 5, 1 and 8 drive ~74% of transactions**, but their ticket sizes differ hugely: Category 1 averages **13.6K**, Category 5 **6.2K**, Category 8 **7.5K**. High-ticket categories such as 10 (19.7K), 7, 6 and 9 are low-volume.
""")

md("### 5.3 Bivariate: purchase amount across categories (box plots)")
code(r"""
cols = ["Gender", "Life_Stage", "Marital_Status", "City_Category", "Stay_In_Current_City_Years", "Occupation"]
fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
for ax, col in zip(axes.flat, cols):
    pal = {"Gender": GENDER_PAL, "Marital_Status": MARITAL_PAL, "Life_Stage": AGE_PAL}.get(col)
    kw = dict(palette=pal) if pal else dict(color=BLUE)
    sns.boxplot(data=df, x=col, y="Purchase", ax=ax, showfliers=False, width=0.55,
                showmeans=True, meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="#0b0b0b", markersize=5), **kw)
    ax.set_title(f"Purchase by {col.replace('_', ' ')}"); ax.set_xlabel("")
    if col == "Occupation": ax.tick_params(axis="x", labelsize=8)
plt.tight_layout(); save(fig, "05_boxplots_bivariate"); plt.show()
""")
code(r"""
fig, ax = plt.subplots(figsize=(15, 4))
sns.boxplot(data=df, x="Product_Category", y="Purchase", hue="Gender", palette=GENDER_PAL,
            showfliers=False, ax=ax, width=0.7, linewidth=0.8)
ax.set_title("Purchase by product category and gender"); ax.set_xlabel("Product category")
ax.legend(title="", loc="upper right", ncol=2)
plt.tight_layout(); save(fig, "06_category_gender_box"); plt.show()
""")
md(r"""
**Observations (bivariate)**
- **Gender:** the male box sits higher (median ~8.1K vs ~7.9K; mean 9.4K vs 8.7K) and has a longer upper whisker.
- **Life stage:** medians are close, but the mean edges up with age; 0-17 is the lowest and 51+ the highest.
- **Marital status:** the two boxes are virtually identical. Marriage does not change how much is spent per item.
- **City:** **City C has the highest per-transaction spend (~9.7K)** despite fewer transactions than City B; City A is the lowest (~8.9K).
- **Stay in city:** no meaningful difference across tenure.
- **Occupation:** small variation (about 8.6K-9.8K); no occupation stands out strongly.
- **Product category is the dominant driver of purchase amount**, far more than any demographic. Within a given category, male and female boxes look very similar. This is the first clue that the gender gap is about *what* people buy.
""")

md("### 5.4 Relationships between variables: crosstabs, heatmaps and pairplot")
code(r"""
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.25]})
ct1 = pd.crosstab(df["Life_Stage"], df["Gender"], normalize="index") * 100
sns.heatmap(ct1, annot=True, fmt=".0f", cmap="Blues", cbar=False, ax=axes[0])
axes[0].set_title("Gender mix within each life stage (%)")
ct2 = pd.crosstab(df["City_Category"], df["Life_Stage"], normalize="index") * 100
sns.heatmap(ct2, annot=True, fmt=".0f", cmap="Blues", cbar=False, ax=axes[1])
axes[1].set_title("Life-stage mix within each city (%)")
top = df["Product_Category"].value_counts().index[:8]
ct3 = (pd.crosstab(df["Product_Category"], df["Gender"], normalize="columns") * 100).loc[top]
sns.heatmap(ct3.T, annot=True, fmt=".1f", cmap="Blues", cbar=False, ax=axes[2])
axes[2].set_title("Share of each gender's transactions by top-8 category (%)")
plt.tight_layout(); save(fig, "07_crosstab_heatmaps"); plt.show()
""")
code(r"""
# Correlation: encode categoricals as ordinal numbers (ordering is natural for age, city tenure)
enc = pd.DataFrame({
    "Gender (M=1)": (df["Gender"] == "Male").astype(int),
    "Age (ordinal)": df["Age"].cat.codes,
    "Occupation": df["Occupation"].astype(int),
    "City (A<B<C)": df["City_Category"].cat.codes,
    "Stay years": df["Stay_In_Current_City_Years"].astype(str).str.replace("+", "", regex=False).astype(int),
    "Married": (df["Marital_Status"] == "Married").astype(int),
    "Product cat.": df["Product_Category"].astype(int),
    "Purchase": df["Purchase"]})
fig, ax = plt.subplots(figsize=(8, 5.2))
sns.heatmap(enc.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, linewidths=1, linecolor="white")
ax.set_title("Correlation heatmap (encoded features)")
plt.tight_layout(); save(fig, "08_correlation_heatmap"); plt.show()
""")
code(r"""
pair_df = enc.sample(3000, random_state=42).assign(Gender=lambda d: d["Gender (M=1)"].map({1: "Male", 0: "Female"}))
g = sns.pairplot(pair_df[["Age (ordinal)", "Occupation", "Product cat.", "Purchase", "Gender"]], hue="Gender",
                 palette=GENDER_PAL, plot_kws=dict(s=8, alpha=0.4, linewidth=0), height=1.9, corner=True)
g.fig.suptitle("Pairplot (random sample of 3,000 transactions)", y=1.02, fontweight="bold")
save(g.fig, "09_pairplot"); plt.show()
""")
md(r"""
**Observations (relationships)**
- **Women are a minority in every life stage** (roughly 23-34% of transactions); the 0-17 group has the highest female share (34%).
- **City C skews older** (more 36-50 and 51+), while **City A and B skew to 26-35**. City C's higher ticket size is consistent with older shoppers spending more per item.
- **Product mix differs by gender:** Category 1 (the high-ticket staple) is **~28% of male transactions vs ~18% of female**, while the lower-ticket Categories 5 and 8 make up **~56% of female transactions vs ~46% of male**.
- **Correlation heatmap:** the only meaningful relationship with Purchase is **Product Category (r ≈ -0.34)**. Gender (r ≈ +0.06), age, city and marital status each have a very weak linear relationship with spend. Age and Marital Status are moderately correlated with each other (+0.31), as expected.
- **Pairplot:** Purchase forms horizontal bands by product category; no demographic variable shows a strong linear trend. Demographics shift the **average** a little, which is precisely what confidence intervals are designed to detect.
""")

# ---------------------------------------------------------------- 6. Gender means
md(r"""
## 6. Gender: are women spending more per transaction than men?
We track every transaction made by female and male customers and compute the average amount spent per transaction.
""")
code(r"""
def describe_groups(col):
    g = df.groupby(col, observed=True)["Purchase"]
    out = g.agg(transactions="size", mean="mean", median="median", std="std", total="sum")
    out["share of revenue %"] = out["total"] / out["total"].sum() * 100
    out["total"] = out["total"] / 1e6
    return out.rename(columns={"total": "revenue (M)"})

gender_stats = describe_groups("Gender")
gender_stats
""")
code(r"""
f, m = (df.loc[df["Gender"] == g, "Purchase"] for g in ["Female", "Male"])
gap = m.mean() - f.mean()
t, pval = stats.ttest_ind(m, f, equal_var=False)
cohen_d = gap / np.sqrt((m.var() + f.var()) / 2)
print(f"Male mean - Female mean = {gap:,.1f}  ({gap / f.mean():.1%} higher for men)")
print(f"Welch t-test: t = {t:.1f}, p-value = {pval:.2e}")
print(f"Effect size (Cohen's d) = {cohen_d:.2f}  -> statistically clear, practically modest")
""")
md(r"""
**Why do men spend more per transaction? A product-mix check.**
If the gap is caused by *what* each gender buys, then re-weighting women's category prices by men's category mix should close most of it.
""")
code(r"""
cat_mean = df.pivot_table(index="Product_Category", columns="Gender", values="Purchase", aggfunc="mean", observed=True)
cat_mix  = pd.crosstab(df["Product_Category"], df["Gender"], normalize="columns")

female_actual       = (cat_mix["Female"] * cat_mean["Female"]).sum()
female_with_men_mix = (cat_mix["Male"]   * cat_mean["Female"]).sum()   # women's prices, men's basket
male_actual         = (cat_mix["Male"]   * cat_mean["Male"]).sum()

decomp = pd.Series({"Female (actual)": female_actual,
                    "Female prices x Male basket mix": female_with_men_mix,
                    "Male (actual)": male_actual})
print(decomp.round(1).to_string())
explained = (female_with_men_mix - female_actual) / (male_actual - female_actual)
print(f"\nShare of the gender gap explained by product mix: {explained:.0%}")
""")
md(r"""
**Answer: No, women are not spending more per transaction than men.**
- The average **female** transaction is **8,734.6** and the average **male** transaction is **9,437.5**, so **men spend about 703 (≈ 8%) more per transaction**.
- The difference is **statistically significant** (Welch t ≈ 46, p ≈ 0), but the effect size is **small (Cohen's d ≈ 0.14)**: it is a real but modest shift, not two different kinds of shopper.
- **Why?** The product-mix check shows that most of the gap comes from **what** men and women buy. Men put a larger share of their basket into high-ticket Category 1 (avg 13.6K), while women lean towards Categories 5 and 8 (avg 6.2K and 7.5K). Within the same category, women and men pay almost exactly the same (e.g. Category 1: 13,597 vs 13,608; Category 8: 7,500 vs 7,499). If women kept their own category prices but bought men's basket mix, their average would be **~9,466**, slightly *above* the male average: product mix explains **the entire gap**.
- Men also make **more transactions per person** and form **72% of customers**, so they generate **77% of Black Friday revenue**.
""")

# ---------------------------------------------------------------- 7. CLT gender
md(r"""
## 7. Gender: Central Limit Theorem and confidence intervals

**Method.**
1. **Bootstrap (CLT experiment):** for each group, draw a random sample of size *n* (with replacement), compute its mean, and repeat **1,000 times**. The 1,000 means form the *sampling distribution of the mean*.
2. We repeat this for **n = 300, 3,000 and 30,000** to see how sample size affects the spread.
3. The **CI** is the central 90% / 95% / 99% of that sampling distribution (percentile method). We cross-check with the analytical CLT formula **mean ± z × σ/√n** on the full data.
""")
code(r"""
SAMPLE_SIZES = [300, 3_000, 30_000]
CONF_LEVELS  = [0.90, 0.95, 0.99]
N_BOOT = 1_000

def bootstrap_means(values, n, reps=N_BOOT, rng=RNG):
    values = np.asarray(values)
    idx = rng.integers(0, len(values), size=(reps, n))
    return values[idx].mean(axis=1)

def ci_table(col, sizes=SAMPLE_SIZES, levels=CONF_LEVELS):
    rows, boot = [], {}
    for grp, vals in df.groupby(col, observed=True)["Purchase"]:
        for n in sizes:
            means = bootstrap_means(vals.values, n)
            boot[(grp, n)] = means
            for cl in levels:
                lo, hi = np.percentile(means, [(1 - cl) / 2 * 100, (1 + cl) / 2 * 100])
                rows.append(dict(group=grp, sample_size=n, confidence=f"{cl:.0%}",
                                 mean_of_means=means.mean(), lower=lo, upper=hi, width=hi - lo))
    return pd.DataFrame(rows), boot

def clt_ci_full(col, levels=CONF_LEVELS):
    rows = []
    for grp, vals in df.groupby(col, observed=True)["Purchase"]:
        se = vals.std(ddof=1) / np.sqrt(len(vals))
        for cl in levels:
            z = stats.norm.ppf((1 + cl) / 2)
            rows.append(dict(group=grp, n=len(vals), confidence=f"{cl:.0%}", mean=vals.mean(),
                             lower=vals.mean() - z * se, upper=vals.mean() + z * se, width=2 * z * se))
    return pd.DataFrame(rows)

def overlap_report(ci_df, size_col="sample_size"):
    out = []
    keys = [size_col, "confidence"] if size_col else ["confidence"]
    for key, sub in ci_df.groupby(keys, sort=False):
        sub = sub.set_index("group")
        grps = list(sub.index)
        for i in range(len(grps) - 1):
            a, b = grps[i], grps[i + 1]
            ov = not (sub.loc[a, "upper"] < sub.loc[b, "lower"] or sub.loc[b, "upper"] < sub.loc[a, "lower"])
            out.append(dict(zip(keys, key if isinstance(key, tuple) else (key,))) | {"pair": f"{a} vs {b}", "overlap": "Yes" if ov else "No"})
    return pd.DataFrame(out)
""")
code(r"""
def plot_clt(boot, groups, palette, title, fname):
    fig, axes = plt.subplots(1, len(SAMPLE_SIZES), figsize=(15, 3.6))
    for ax, n in zip(axes, SAMPLE_SIZES):
        for g in groups:
            sns.histplot(boot[(g, n)], bins=40, kde=True, stat="density", color=palette[g], alpha=0.35,
                         edgecolor="white", linewidth=0.2, ax=ax, label=g)
        ax.set_title(f"Sample size n = {n:,}"); ax.set_xlabel("Sample mean of Purchase"); ax.set_ylabel("")
    axes[0].legend(fontsize=8)
    fig.suptitle(title, fontweight="bold", y=1.03)
    plt.tight_layout(); save(fig, fname); plt.show()

def plot_ci(ci_df, groups, palette, title, fname, size_col="sample_size"):
    sizes = ci_df[size_col].unique() if size_col else [None]
    fig, axes = plt.subplots(1, len(sizes), figsize=(15, 0.55 * len(groups) * 3 + 1), sharey=True, squeeze=False)
    for ax, n in zip(axes.flat, sizes):
        sub = ci_df if n is None else ci_df[ci_df[size_col] == n]
        ypos = 0
        ticks, labels = [], []
        for g in groups:
            for cl in ["90%", "95%", "99%"]:
                r = sub[(sub["group"] == g) & (sub["confidence"] == cl)].iloc[0]
                centre = (r["lower"] + r["upper"]) / 2
                ax.errorbar(centre, ypos, xerr=[[centre - r["lower"]], [r["upper"] - centre]], fmt="o",
                            color=palette[g], capsize=4, lw=2, ms=5)
                ticks.append(ypos); labels.append(f"{g} {cl}"); ypos += 1
            ypos += 0.8
        ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=8); ax.invert_yaxis()
        ax.set_title("Full data (analytical CLT)" if n is None else f"n = {n:,}")
        ax.set_xlabel("Average purchase per transaction")
    fig.suptitle(title, fontweight="bold", y=1.02)
    plt.tight_layout(); save(fig, fname); plt.show()
""")
code(r"""
GENDERS = ["Female", "Male"]
gender_ci, gender_boot = ci_table("Gender")
plot_clt(gender_boot, GENDERS, GENDER_PAL, "CLT in action: distribution of the sample mean, Female vs Male", "10_clt_gender")
""")
code(r"""
# Normality check: are the bootstrap means normal even though raw purchases are not?
fig, axes = plt.subplots(1, 3, figsize=(15, 3.2))
stats.probplot(df.loc[df["Gender"] == "Female", "Purchase"].sample(3000, random_state=1), dist="norm", plot=axes[0])
axes[0].set_title("Q-Q: raw female purchases (not normal)")
stats.probplot(gender_boot[("Female", 300)], dist="norm", plot=axes[1]);  axes[1].set_title("Q-Q: female sample means, n=300")
stats.probplot(gender_boot[("Female", 30000)], dist="norm", plot=axes[2]); axes[2].set_title("Q-Q: female sample means, n=30,000")
for ax in axes:
    ax.get_lines()[0].set(markerfacecolor=ORANGE, markeredgecolor=ORANGE, markersize=3); ax.get_lines()[1].set(color="#0b0b0b")
plt.tight_layout(); save(fig, "11_qq_normality"); plt.show()
""")
code(r"""
gender_ci.pivot_table(index=["sample_size", "confidence"], columns="group", values=["lower", "upper", "width"]).round(1)
""")
code(r"""
plot_ci(gender_ci, GENDERS, GENDER_PAL, "Confidence intervals of average spend: Female vs Male, by sample size and confidence level", "12_ci_gender")
overlap_report(gender_ci).pivot(index="sample_size", columns="confidence", values="overlap")
""")
code(r"""
gender_full = clt_ci_full("Gender")
display(gender_full.round(1))
overlap_report(gender_full, size_col=None)
""")
code(r"""
# How CI width shrinks with sample size (95%)
fig, ax = plt.subplots(figsize=(15, 3.4))
w = gender_ci[gender_ci["confidence"] == "95%"]
for g in GENDERS:
    s = w[w["group"] == g]
    ax.plot(s["sample_size"], s["width"], marker="o", lw=2, color=GENDER_PAL[g], label=g)
    for _, r in s.iterrows():
        ax.annotate(f"{r['width']:,.0f}", (r["sample_size"], r["width"]), textcoords="offset points", xytext=(6, 6), fontsize=8)
ax.set_xscale("log"); ax.set_xlabel("Sample size (log scale)"); ax.set_ylabel("95% CI width")
ax.set_title("Bigger samples give narrower intervals: width shrinks by about √10 ≈ 3.2× for each 10× more data")
ax.legend(); plt.tight_layout(); save(fig, "13_ci_width_vs_n"); plt.show()
""")
md(r"""
**Observations: CLT and confidence intervals for gender**
- **The CLT holds:** raw purchases are skewed and multi-modal, yet the bootstrap means form a clean **bell curve** at every sample size (Q-Q points sit on the line). The centre of each bell curve matches the group mean.
- **Sample size controls precision.** As *n* goes from 300 to 3,000 to 30,000, the spread of the sample means shrinks by about √10 each time, and the **95% CI narrows from ~1,000-1,150 to ~350-370 to ~110**.
- **Confidence level controls width.** For a given *n*, the 99% CI is the widest and the 90% CI the narrowest. Higher confidence costs precision.
- **Overlap:**
  - At **n = 300** the male and female intervals **overlap** at all three confidence levels: a small sample cannot tell the groups apart.
  - From **n = 3,000** upward the intervals **do not overlap**, even at 99%.
  - Using **all the data** (analytical CLT), the 95% intervals are **Female [8,709, 8,760]** and **Male [9,422, 9,453]**, with a gap of more than 650 between them.
- **Conclusion for the 50M + 50M population:** we can be 95% (and even 99%) confident that the **average male transaction is higher than the average female transaction**. Women do **not** spend more per transaction.

**How Walmart can use this**
- Because the intervals do not overlap, the difference is **real and repeatable**, so gender-specific strategies are justified.
- Since the gap comes mostly from **basket mix**, the highest-value action is to **lift women's spend in high-ticket categories** (e.g. Category 1, 6, 10) through targeted bundles, curated "Black Friday picks" and personalised offers.
- The female customer base is **under-represented (28% of shoppers)** even though the population is 50/50: there is a large **acquisition opportunity**.
""")

# ---------------------------------------------------------------- 8. Marital
md("## 8. Marital status: Married vs Unmarried")
code(r"""
describe_groups("Marital_Status")
""")
code(r"""
MARITAL = ["Unmarried", "Married"]
marital_ci, marital_boot = ci_table("Marital_Status")
plot_clt(marital_boot, MARITAL, MARITAL_PAL, "CLT: distribution of the sample mean, Unmarried vs Married", "14_clt_marital")
""")
code(r"""
plot_ci(marital_ci, MARITAL, MARITAL_PAL, "Confidence intervals of average spend: Unmarried vs Married", "15_ci_marital")
marital_full = clt_ci_full("Marital_Status")
display(marital_full.round(1))
display(overlap_report(marital_ci).pivot(index="sample_size", columns="confidence", values="overlap"))
u, mr = (df.loc[df["Marital_Status"] == g, "Purchase"] for g in MARITAL)
print(f"Welch t-test: p-value = {stats.ttest_ind(u, mr, equal_var=False).pvalue:.3f}")
""")
md(r"""
**Observations: Married vs Unmarried**
- Average spend per transaction is **9,265.9 (Unmarried)** vs **9,261.2 (Married)**: a difference of under 5.
- The sampling distributions **lie on top of each other** at every sample size, and the confidence intervals **overlap at every sample size and every confidence level** (full-data 95%: Unmarried [9,249, 9,283], Married [9,241, 9,282]). The t-test agrees (p ≈ 0.74).
- **Conclusion:** for the population, **marital status has no measurable effect on spend per transaction.**
- **How Walmart can use this:** do not split Black Friday pricing or discount depth by marital status. Instead use marital status for **what** to promote (e.g. home and family bundles for married shoppers, lifestyle and tech for unmarried shoppers), since the *ticket size* is the same.
""")

# ---------------------------------------------------------------- 9. Age
md(r"""
## 9. Age: life-stage bins (0-17, 18-25, 26-35, 36-50, 51+)
The original seven age bins are merged into the five life stages suggested in the case: 36-45 and 46-50 become **36-50**, and 51-55 and 55+ become **51+**.
""")
code(r"""
describe_groups("Life_Stage")
""")
code(r"""
age_ci, age_boot = ci_table("Life_Stage")
plot_clt(age_boot, AGE_ORDER, AGE_PAL, "CLT: distribution of the sample mean by life stage", "16_clt_age")
""")
code(r"""
age_full = clt_ci_full("Life_Stage")
plot_ci(age_full.assign(sample_size=None), AGE_ORDER, AGE_PAL,
        "Confidence intervals of average spend by life stage (full data, analytical CLT)", "17_ci_age", size_col=None)
display(age_full.round(1))
overlap_report(age_full, size_col=None).pivot(index="pair", columns="confidence", values="overlap").loc[[f"{a} vs {b}" for a, b in zip(AGE_ORDER, AGE_ORDER[1:])]]
""")
code(r"""
print("Overlap of neighbouring age groups in bootstrap CIs, by sample size:")
age_ov = overlap_report(age_ci)
age_ov.pivot_table(index=["pair"], columns=["sample_size", "confidence"], values="overlap", aggfunc="first").loc[[f"{a} vs {b}" for a, b in zip(AGE_ORDER, AGE_ORDER[1:])]]
""")
code(r"""
f_stat, p_anova = stats.f_oneway(*[g.values for _, g in df.groupby("Life_Stage", observed=True)["Purchase"]])
print(f"One-way ANOVA across life stages: F = {f_stat:.1f}, p-value = {p_anova:.2e}")
""")
md(r"""
**Observations: Age**
- Average spend per transaction **rises with age**: 0-17 **8,933** → 18-25 **9,170** → 26-35 **9,253** → 36-50 **9,295** → 51+ **9,464**. The 51+ group spends about **530 (6%) more per item** than teenagers.
- With **small samples (n = 300 and 3,000)**, every neighbouring pair overlaps: the age effect is invisible. At **n = 30,000**, 0-17 and 51+ separate from their neighbours, while the middle groups still overlap. (Note: 0-17 has only 15,102 transactions, so its n = 30,000 bootstrap re-uses rows; the full-data CI below is the reliable one for that group.)
- With **all the data**:
  - **0-17** is clearly below everyone else, and **51+** is clearly above everyone else, at every confidence level.
  - **18-25 vs 26-35** do **not** overlap (although they come close at 99%).
  - **26-35 vs 36-50** is the borderline pair: **no overlap at 90%, but overlap at 95% and 99%**. This shows how the business's choice of confidence level can change the conclusion. These two groups are best treated as **one "prime-age" segment**.
- ANOVA confirms that age groups differ overall (p ≈ 0).
- **How Walmart can use this:** the **26-50** age band brings the most transactions and revenue (~68% of transactions), so it is the core audience. The **51+** group has the **highest ticket size**, making it ideal for premium and high-value offers. **0-17** has the lowest ticket and is best reached through family / parent-led promotions.
""")

# ---------------------------------------------------------------- 10. Customer level
md(r"""
## 10. Bonus: the customer-level view (total spend per shopper)
Per-transaction averages answer "how big is each purchase?". Management may also ask "how much does each **customer** spend in total on Black Friday?". Here each of the 5,891 customers is one observation.
""")
code(r"""
def customer_ci(col, cl=0.95):
    rows = []
    for g, v in users.groupby(col, observed=True)["Total_Spend"]:
        se = v.std(ddof=1) / np.sqrt(len(v)); z = stats.norm.ppf((1 + cl) / 2)
        rows.append(dict(segment=col, group=g, customers=len(v), mean=v.mean(), lower=v.mean() - z * se, upper=v.mean() + z * se))
    return pd.DataFrame(rows)

cust_ci = pd.concat([customer_ci(c) for c in ["Gender", "Marital_Status", "Life_Stage"]], ignore_index=True)
display(cust_ci.round(0))

fig, ax = plt.subplots(figsize=(15, 3.8))
pal = GENDER_PAL | MARITAL_PAL | AGE_PAL
for i, r in cust_ci.iterrows():
    ax.errorbar(r["mean"] / 1e3, i, xerr=[[(r["mean"] - r["lower"]) / 1e3], [(r["upper"] - r["mean"]) / 1e3]],
                fmt="o", color=pal[r["group"]], capsize=4, lw=2)
ax.set_yticks(range(len(cust_ci))); ax.set_yticklabels(cust_ci["segment"].str.replace("_", " ") + ": " + cust_ci["group"].astype(str), fontsize=8)
ax.invert_yaxis(); ax.set_xlabel("Average total Black Friday spend per customer (thousands)")
ax.set_title("95% CI of total spend per customer")
plt.tight_layout(); save(fig, "18_customer_level_ci"); plt.show()
""")
md(r"""
**Observations: customer level**
- **Gender:** an average man spends **~925K** over Black Friday vs **~712K** for a woman. The 95% CIs (**[896K, 955K]** vs **[673K, 751K]**) **do not overlap**, so the conclusion is the same as per transaction, and stronger: men buy slightly bigger tickets **and** buy more items.
- **Marital status:** intervals overlap (unmarried ~881K vs married ~844K). No significant difference.
- **Age:** the **26-35** shopper is the most valuable customer (~990K), significantly ahead of 51+ (~666K) and 0-17 (~619K). The 51+ group buys **fewer but pricier** items, while 26-35 buy **many** items.
""")

# ---------------------------------------------------------------- Summary table
code(r"""
summary_ci = pd.concat([clt_ci_full(c).assign(segment=c) for c in ["Gender", "Marital_Status", "Life_Stage"]])
summary_ci = summary_ci[["segment", "group", "n", "confidence", "mean", "lower", "upper", "width"]]
os.makedirs("../results", exist_ok=True)
summary_ci.round(2).to_csv("../results/confidence_intervals_summary.csv", index=False)
gender_ci.round(2).to_csv("../results/bootstrap_ci_gender.csv", index=False)
marital_ci.round(2).to_csv("../results/bootstrap_ci_marital.csv", index=False)
age_ci.round(2).to_csv("../results/bootstrap_ci_age.csv", index=False)
cust_ci.round(2).to_csv("../results/customer_level_ci.csv", index=False)
summary_ci[summary_ci["confidence"] == "95%"].round(1).reset_index(drop=True)
""")

# ---------------------------------------------------------------- 11. Insights
md(r"""
## 11. Final insights

**Distribution of variables and relationships**

1. **Purchase is right-skewed and multi-modal** (mean 9,264 > median 8,047). The peaks come from product-category price bands, so **product category is the strongest driver of spend** (correlation ≈ -0.34). Demographics shift the average only modestly.
2. **The shopper base is skewed:** 75% of transactions and 72% of customers are male; 40% of transactions come from ages 26-35; 59% are unmarried; City B has the most transactions but **City C has the highest ticket size**.
3. **Outliers (0.49%) are real high-value purchases** from premium categories and were kept.

**Generalising to the population (CLT and confidence intervals)**

4. **Gender:** men spend **~703 (8%) more per transaction** than women. The CIs do not overlap at 90, 95 or 99% (full data), so this holds for the 50M + 50M population. **Women do not spend more per transaction.**
5. **Why:** the gender gap is explained entirely by **basket mix**: men buy more of high-ticket Category 1, women more of mid/low-ticket Categories 5 and 8. Within the same category, women pay the same as men.
6. **Marital status makes no difference** to spend per transaction: the CIs overlap at every sample size and confidence level.
7. **Age matters, modestly:** spend per transaction rises with age. 0-17 is clearly lowest and 51+ clearly highest; 26-35 and 36-50 are statistically almost identical (overlap at 95%/99%).
8. **Sample size matters:** with n = 300 none of the gender, marital or age differences are detectable. The gender gap appears from n = 3,000; the smaller age differences need tens of thousands of observations. Walmart should rely on large samples before making segment-level decisions.
9. **Confidence level is a business trade-off:** a 99% interval is safer but wider; 90% is tighter but riskier. The 26-35 vs 36-50 comparison flips between "different" (90%) and "same" (95%), showing that borderline differences should not drive big decisions.
10. **Customer-level view confirms it:** a male customer spends ~925K vs ~712K for a female customer over Black Friday, and the 26-35 shopper is the most valuable customer overall.
""")

# ---------------------------------------------------------------- 12. Recommendations
md(r"""
## 12. Recommendations for Walmart

*Simple, actionable steps, no jargon.*

1. **Win more women shoppers.** Women are half the population but only about a quarter of Black Friday shoppers. Run women-focused Black Friday campaigns, early-access sales and referral rewards to bring more of them into stores.
2. **Help women discover bigger-ticket products.** Women already spend well, but in lower-price categories. Create "Top Black Friday picks" bundles that pair their favourite categories (5 and 8) with higher-value items from Categories 1, 6 and 10, and offer "spend more, save more" deals.
3. **Keep male shoppers loyal.** Men bring 77% of Black Friday revenue. Protect it with loyalty points, member-only deals and stock guarantees on the high-ticket items they buy most.
4. **Do not change prices based on marital status.** Married and unmarried customers spend the same per item. Use marital status only to choose *which* products to show (family and home bundles vs lifestyle and gadgets).
5. **Focus the main campaign on ages 26-50.** They make two-thirds of all purchases. Treat 26-35 and 36-50 as one group in planning.
6. **Offer premium deals to the 51+ group.** They have the highest spend per item. Promote quality, durable and premium products with easy in-store help and delivery.
7. **Reach teenagers through parents.** The 0-17 group spends the least per item. Bundle youth products into family offers instead of targeting teens directly.
8. **Invest more in City C, grow City A.** City C shoppers spend the most per item, so stock premium ranges there. City A has the lowest ticket size and could benefit from upselling and bundles.
9. **Stock and staff for the big three categories.** Categories 1, 5 and 8 make up about three in four transactions. Make sure they never run out during Black Friday.
10. **Measure before acting.** Small samples hide real differences. Test new offers on large groups (thousands of customers) and use 95% confidence as the standard for big decisions.
""")

md(r"""
---
*Data: Walmart Black Friday transactions (550,068 rows). Full code, data, images and results: see the project repository README.*
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, "notebooks/Walmart_CLT_Case_Study.ipynb")
print("written", len(cells), "cells")
