# Walmart Black Friday: Confidence Interval and CLT – Insights and Recommendations

**PDF of the notebook:** <paste your Google Drive link here>
**Full project (code, data, charts):** <paste your GitHub repo link here>

## Key insights

**1. Data overview**
- 550,068 transactions from 5,891 customers and 3,631 products. No missing values, no duplicates.
- Purchase amount ranges from 12 to 23,961. Mean 9,264 is higher than median 8,047, so the data is right-skewed.
- 0.49% of transactions are outliers (above 21,400). They are genuine premium purchases (mostly Product Category 10), so they were kept.
- 75% of transactions and 72% of customers are male. 40% of transactions come from ages 26-35. 59% are from unmarried customers. City B has the most transactions; City C has the highest spend per transaction.
- Product category is the strongest driver of purchase amount. Categories 1, 5 and 8 make up ~74% of all transactions.

**2. Are women spending more per transaction than men? No.**
- Average spend per transaction: Male 9,437.5 vs Female 8,734.6. Men spend about 703 (8%) more.
- The difference is statistically significant (Welch t-test p ≈ 0) but modest in size (Cohen's d = 0.14).
- Why: the gap comes entirely from product mix. Men buy more of high-ticket Category 1 (average 13.6K); women buy more of Categories 5 and 8 (6.2K and 7.5K). Within the same category, women and men pay the same amount. If women bought men's category mix at their own prices, their average would be ~9,466, slightly above men.

**3. Confidence intervals for gender (CLT)**
- Raw purchases are not normal, but the means of repeated samples form a normal curve (the CLT), which lets us build confidence intervals.
- Full data, 95% CI: Female [8,709, 8,760], Male [9,422, 9,453]. The intervals do not overlap at 90%, 95% or 99%.
- Sample size matters: at n = 300 the male and female intervals overlap; from n = 3,000 upward they do not. The 95% CI width shrinks from ~1,000 (n = 300) to ~350 (n = 3,000) to ~110 (n = 30,000).
- Confidence level matters: 99% intervals are wider than 95%, which are wider than 90%.
- Conclusion: for the population of 50M men and 50M women, the average male spend per transaction is higher than the female average.

**4. Married vs Unmarried**
- Unmarried 9,265.9 vs Married 9,261.2. The confidence intervals overlap at every sample size and every confidence level (t-test p = 0.73).
- Marital status has no effect on spend per transaction.

**5. Age (life stages)**
- Spend per transaction rises with age: 0-17: 8,934 | 18-25: 9,170 | 26-35: 9,253 | 36-50: 9,295 | 51+: 9,464.
- 0-17 is clearly the lowest and 51+ clearly the highest (no overlap with neighbours at any confidence level).
- 26-35 and 36-50 are borderline: they do not overlap at 90% but do overlap at 95% and 99%. They behave like one segment.
- With small samples (n = 300 or 3,000) none of the age differences are visible.

**6. Customer-level view (total spend per customer)**
- An average male customer spends ~925K over Black Friday vs ~712K for a female customer (95% CIs do not overlap).
- The 26-35 customer is the most valuable (~990K), buying many items. The 51+ customer buys fewer but pricier items.

## Recommendations

1. **Win more women shoppers.** Women are half the population but only about a quarter of Black Friday shoppers. Run women-focused campaigns, early-access sales and referral rewards.
2. **Help women discover bigger-ticket products.** Create "Top Black Friday picks" bundles that pair Categories 5 and 8 with higher-value items from Categories 1, 6 and 10, plus "spend more, save more" deals.
3. **Keep male shoppers loyal.** Men bring 77% of Black Friday revenue. Use loyalty points, member-only deals and stock guarantees on the high-ticket items they buy.
4. **Do not change prices based on marital status.** Married and unmarried customers spend the same per item. Use marital status only to choose which products to show (family and home bundles vs lifestyle and gadgets).
5. **Focus the main campaign on ages 26-50.** They make two-thirds of purchases. Plan for 26-35 and 36-50 as one group.
6. **Offer premium deals to customers aged 51+.** They spend the most per item. Promote quality and premium products with in-store help and delivery.
7. **Reach teenagers through parents.** The 0-17 group spends the least per item. Bundle youth products into family offers.
8. **Stock premium ranges in City C and upsell in City A.** City C shoppers spend the most per item; City A spends the least.
9. **Never run out of Categories 1, 5 and 8.** They make up about three in four transactions.
10. **Test on large groups before deciding.** Small samples hide real differences. Use thousands of customers and 95% confidence as the standard for big decisions.
