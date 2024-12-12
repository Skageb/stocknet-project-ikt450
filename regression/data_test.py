import os
import pandas as pd

# Define the folder path
folder_path = "./dataset/price/preprocessed/"

# Initialize a dictionary to store issues for each file
issues = {}

# Loop through all files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".txt"):  # Ensure it's a data file
        file_path = os.path.join(folder_path, file_name)
        company_name = file_name.split(".")[0]  # Extract company name

        # Load the data
        try:
            df = pd.read_csv(file_path, sep="\t", header=None)
            df.columns = ["date", "movement_percent", "open_price", "high_price", "low_price", "close_price", "volume"]
            df["date"] = pd.to_datetime(df["date"])  # Convert date to datetime

            # Check for missing values
            missing_values = df.isnull().sum()
            missing_dates = df["date"].isnull().sum()
            total_missing = missing_values.sum()

            # Check for missing or empty features
            empty_features = (df == 0).sum()

            # Detect missing dates (gaps in time series)
            date_range = pd.date_range(start=df["date"].min(), end=df["date"].max())
            missing_dates_in_range = len(set(date_range) - set(df["date"]))

            # Store issues for this company
            issues[company_name] = {
                "missing_values": total_missing,
                "missing_dates_column": missing_dates,
                "empty_features": empty_features.to_dict(),
                "missing_dates_in_range": missing_dates_in_range,
            }

        except Exception as e:
            issues[company_name] = {"error": str(e)}

# Save results to a file
output_path = "data_issues_report.txt"
with open(output_path, "w") as f:
    for company, issue in issues.items():
        f.write(f"Company: {company}\n")
        for key, value in issue.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")

print(f"Data issues report saved to {output_path}")
