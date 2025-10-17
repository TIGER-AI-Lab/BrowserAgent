import pandas as pd
import pprint

data_path = "/home/zhiheng/cogito/verl-tool/data/wikiQA_debug/test.parquet"
# data_path = "/home/zhiheng/cogito/verl-tool/data/acecoder/AceCoderV2-mini-processed-with-execution-prompt/test.parquet"
df = pd.read_parquet(data_path)
row = df.iloc[0]

pp = pprint.PrettyPrinter(indent=2, width=120)

for col in df.columns:
    print(f"==== {col} ====")
    val = row[col]
    if isinstance(val, (dict, list)):
        pp.pprint(val)
    else:
        print(val)
    print()
