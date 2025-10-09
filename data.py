import pandas as pd


def load_debt_data():
    df = pd.read_csv('data/debt_92-05.csv', sep=';')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.rename(columns={'1_variable_attribute_label': 'state'})
    df['time'] = pd.to_datetime(df['time']).dt.year
    return df

def total_annual_debt_by_states(df, state_names):
    filtered_df = df[df['state'].isin(state_names)]
    filtered_df = filtered_df[filtered_df['2_variable_attribute_label'] == "Länder"]
    filtered_df['total_annual_debt'] = filtered_df.groupby(['state', 'time'])['value'].transform('sum')
    result_df = filtered_df.groupby(['state', 'time'])['total_annual_debt'].agg(lambda x: list(x)[0]).reset_index()
    return result_df

def filter_by_year(df, year):
    return df[df['time'] == year]