import pandas as pd
import sys

data = pd.read_pickle(sys.argv[1])
print(data.info(verbose=True))

# for column in data.columns.tolist():
#     print(data.groupby([column]).agg(['max']))