import pandas as pd

def main():
    df = pd.read_csv('synthetic_data_1.csv')
    df.columns = df.columns.str.lstrip()
    df.to_csv('synthetic_data_1.csv', index=False)

if __name__ == "__main__":
    main()