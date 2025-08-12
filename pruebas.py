import pandas as pd

data = {'Category': ['A', 'B', 'A', 'C', 'B', 'A'],
        'Value': [10, 15, 20, 5, 25, 30]}
df = pd.DataFrame(data)

# Group by 'Category' and calculate the sum of 'Value' for each category
grouped_data = df.groupby('Category')['Value'].sum()

print(grouped_data)