## School Performance & Household Income

Coming from a background in education I decided to investigate whether
school performance could be predicted by the income of the community. I know
that even within a single district, funding can vary substantially as well as
student achievement. This achievement gap is a concern that I decided to try
tackling with data science regression techniques. I figured that if my findings
indicate wealthier communities perform better on California state testing, then
the CA Department of Education could make better informed decisions about allocation
of funding.

My approach to this problem was to use [2018 data](https://caaspp.cde.ca.gov/sb2018/ResearchFileList)
from Smarter Balanced Assessment Consortium test data obtained from CA
Assessment of Student Performance and Progress. I then paired zip codes for each
school to median household income using [United States Zip Codes](https://www.unitedstateszipcodes.org/).

Once all the data was collected, I decided upon using Root Mean Square Error as
my metric of model performance. After feature engineering my data, I trained
and validated my data using Linear, Ridge, LASSO, and Polynomial Regressions.
Comparing the model results, I selected Linear Regression even though it performed
worse than Ridge (only marginally) since it is a simpler model.

Despite my best attempts to eek out the best results while still keeping
Durbin-Watson, Kurtosis, Skew, and Condition Number in check, my model's
prediction of income are roughly 30% off from the true median income. So while
there is some predictive power, I concluded that Department of Education should
probably not use this as a metric of how to fund schools as it might in some cases
actually compound achievement gap issues by under funding some schools that perform
poorly and vice-versa.

For more details of my implementation, please visit my [repository](https://github.com/MattEding/ProjectLuther).
