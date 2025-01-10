import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Function to clean salary columns
def clean_salary(salary):
    return pd.to_numeric(salary.replace(',', ''), errors='coerce')

# Load the dataset
file_path = 'SIMA-Dataset.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# Clean the salary columns
df['Average'] = df['Average'].apply(clean_salary)
df['Lowest'] = df['Lowest'].apply(clean_salary)
df['Highest'] = df['Highest'].apply(clean_salary)

# Remove data with frequency other than /yr
df = df[df['yr/mo/hr'] == '/yr']

# Drop rows with NaN values in salary columns
df_clean = df.dropna(subset=['Average', 'Lowest', 'Highest'])

# Remove outliers using IQR method
Q1 = df_clean['Average'].quantile(0.25)
Q3 = df_clean['Average'].quantile(0.75)
IQR = Q3 - Q1
df_clean = df_clean[(df_clean['Average'] >= (Q1 - 1.5 * IQR)) & (df_clean['Average'] <= (Q3 + 1.5 * IQR))]

# Calculate the population mean and standard deviation
population_mean = df_clean['Average'].mean()
population_std = df_clean['Average'].std()
print(f"Population Mean Average Salary: {population_mean}")
print(f"Population Standard Deviation: {population_std}")

# Randomly select 29 samples
df_sample = df_clean.sample(n=29, random_state=42)

# Calculate the sample mean and standard deviation
sample_mean = df_sample['Average'].mean()
sample_std = df_sample['Average'].std()
print(f"Sample Mean Average Salary: {sample_mean}")
print(f"Sample Standard Deviation: {sample_std}")

# Perform two-tailed t-test against the population mean
t_stat_two_tailed, p_value_two_tailed = stats.ttest_1samp(df_sample['Average'], population_mean)

# Set alpha value
alpha = 0.05

# Print the results for two-tailed test
print("\nTwo-tailed Hypothesis Testing for Sample vs. Population Mean:")
print("Null Hypothesis (H0): The sample mean is equal to the population mean.")
print("Alternative Hypothesis (H1): The sample mean is not equal to the population mean.")
print("T-statistic:", t_stat_two_tailed)
print("P-value:", p_value_two_tailed)

if p_value_two_tailed < alpha:
    print("Reject the null hypothesis: The sample mean significantly differs from the population mean.")
else:
    print("Accept the null hypothesis: The sample mean does not significantly differ from the population mean.")

# Perform one-tailed t-test against the population mean (greater)
t_stat_one_tailed_greater, p_value_one_tailed_greater = stats.ttest_1samp(df_sample['Average'], population_mean, alternative='greater')

# Print the results for one-tailed test (greater)
print("\nOne-tailed Hypothesis Testing for Sample vs. Population Mean (Greater):")
print("Null Hypothesis (H0): The sample mean is less than or equal to the population mean.")
print("Alternative Hypothesis (H1): The sample mean is greater than the population mean.")
print("T-statistic:", t_stat_one_tailed_greater)
print("P-value:", p_value_one_tailed_greater)

if p_value_one_tailed_greater < alpha:
    print("Reject the null hypothesis: The sample mean is significantly greater than the population mean.")
else:
    print("Accept the null hypothesis: The sample mean is not significantly greater than the population mean.")

# Perform one-tailed t-test against the population mean (less)
t_stat_one_tailed_less, p_value_one_tailed_less = stats.ttest_1samp(df_sample['Average'], population_mean, alternative='less')

# Print the results for one-tailed test (less)
print("\nOne-tailed Hypothesis Testing for Sample vs. Population Mean (Less):")
print("Null Hypothesis (H0): The sample mean is greater than or equal to the population mean.")
print("Alternative Hypothesis (H1): The sample mean is less than the population mean.")
print("T-statistic:", t_stat_one_tailed_less)
print("P-value:", p_value_one_tailed_less)

if p_value_one_tailed_less < alpha:
    print("Reject the null hypothesis: The sample mean is significantly less than the population mean.")
else:
    print("Accept the null hypothesis: The sample mean is not significantly less than the population mean.")

# Display normal distribution curve for sample and population
plt.figure(figsize=(14, 7))

# Plot population distribution
sns.histplot(df_clean['Average'], kde=True, color='blue', label='Population', stat='density', linewidth=0)
# Plot sample distribution
sns.histplot(df_sample['Average'], kde=True, color='orange', label='Sample', stat='density', linewidth=0)

plt.title('Normal Distribution of Average Salaries')
plt.xlabel('Average Salary (INR)')
plt.ylabel('Density')
plt.legend()
plt.show()

# Scatter plot: Rating vs. Average Salary for Total Population
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df_clean, x='Rating', y='Average', hue='Company', palette='viridis', s=100)
plt.axhline(y=df_clean['Average'].mean(), color='red', linestyle='--', label='Population Mean Salary')
plt.title('Company Ratings vs. Average Salaries (Total Population)')
plt.xlabel('Rating')
plt.ylabel('Average Salary (INR)')
plt.legend()
plt.show()

# Scatter plot: Rating vs. Average Salary for Sample
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df_sample, x='Rating', y='Average', hue='Company', palette='viridis', s=100)
plt.axhline(y=df_sample['Average'].mean(), color='red', linestyle='--', label='Sample Mean Salary')
plt.title('Company Ratings vs. Average Salaries (Sample)')
plt.xlabel('Rating')
plt.ylabel('Average Salary (INR)')
plt.legend()
plt.show()

# Suggest companies for freshers based on salary expectations
expected_salary = float(input("Enter your expected salary (INR): "))

suggested_companies = df_clean[df_clean['Average'] >= expected_salary][['Company', 'Rating', 'Average']]
print("\nSuggested Companies based on your expected salary:")
print(suggested_companies)

# Perform hypothesis testing based on the expected salary
t_stat, p_value = stats.ttest_1samp(df_clean['Average'], expected_salary)

print("\nHypothesis Testing for Expected Salary:")
print(f"Expected Salary: {expected_salary}")
print("Null Hypothesis (H0): The population mean is equal to the expected salary.")
print("Alternative Hypothesis (H1): The population mean is not equal to the expected salary.")
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

if p_value < alpha:
    print("Reject the null hypothesis: The population mean significantly differs from the expected salary.")
else:
    print("Accept the null hypothesis: The population mean does not significantly differ from the expected salary.")

# Display dataset info
print(df_clean.info())
print(df_clean.describe())

# Conclusion and analysis
print("\nConclusion and Analysis:")
print("Based on the t-test for sample vs. population mean, we have the following results:")
print(f"Two-tailed T-statistic: {t_stat_two_tailed}, P-value: {p_value_two_tailed}")
if p_value_two_tailed < alpha:
    print("We reject the null hypothesis, indicating a significant difference between the sample mean and the population mean.")
else:
    print("We accept the null hypothesis, indicating no significant difference between the sample mean and the population mean.")

print("\nBased on the one-tailed tests for sample vs. population mean, we have the following results:")
print(f"One-tailed (greater) T-statistic: {t_stat_one_tailed_greater}, P-value: {p_value_one_tailed_greater}")
if p_value_one_tailed_greater < alpha:
    print("We reject the null hypothesis, indicating the sample mean is significantly greater than the population mean.")
else:
    print("We accept the null hypothesis, indicating the sample mean is not significantly greater than the population mean.")

print(f"One-tailed (less) T-statistic: {t_stat_one_tailed_less}, P-value: {p_value_one_tailed_less}")
if p_value_one_tailed_less < alpha:
    print("We reject the null hypothesis, indicating the sample mean is significantly less than the population mean.")
else:
    print("We accept the null hypothesis, indicating the sample mean is not significantly less than the population mean.")

print("\nBased on the t-test for expected salary vs. population mean, we have the following results:")
print(f"T-statistic: {t_stat}, P-value: {p_value}")
if p_value < alpha:
    print("We reject the null hypothesis, indicating a significant difference between the population mean and the expected salary.")
else:
    print("We accept the null hypothesis, indicating no significant difference between the population mean and the expected salary.")

# Practical advice for freshers
print("\nPractical Advice for Freshers:")
print("1. Look for companies that have a combination of a high average salary and a high rating.")
print("2. Do not focus solely on the salary; consider the company's rating as well. High ratings often correlate with better work-life balance and job satisfaction.")
print("3. Use the scatter plot to identify companies that stand out in terms of both salary and rating.")
print("4. Choose companies that not only pay well but are also highly rated by employees.")
