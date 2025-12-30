import pandas as pd
try:
    df = pd.DataFrame({'A': ['2021-01-01', '2021-01-02']})
    print("Attempting diff on strings...")
    df['A'].diff()
except Exception as e:
    print(f"Error: {e}")
