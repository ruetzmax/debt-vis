import pandas as pd


def load_debt_data():
    df = pd.read_csv('data/debt_92-05.csv', sep=';')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.rename(columns={'1_variable_attribute_label': 'state'})
    df['time'] = pd.to_datetime(df['time']).dt.year
    df = df.rename(columns={'time': 'year'})
    return df

def total_annual_debt():
    df = load_debt_data()
    df = df[df['2_variable_attribute_label'] == "Länder"]
    df['value'] = df.groupby(['state', 'year'])['value'].transform('sum')
    result_df = df.groupby(['state', 'year'])['value'].agg(lambda x: list(x)[0]).reset_index()
    return result_df

def total_annual_unemployment():
    df = pd.read_csv('data/unemployment_91-24.csv', sep=';')
    df = df[df['2_variable_attribute_label'] == "Insgesamt"]
    df = df.rename(columns={'1_variable_attribute_label': 'state'})
    df = df.rename(columns={'time': 'year'})
    df = df.groupby(['state', 'year'])['value'].agg(lambda x: list(x)[0]).reset_index()
    return df

def filter_by_year(df, year):
    return df[df['year'] == year]

def filter_by_years(df, min_year, max_year):
    return df[(df['year'] >= min_year) & (df['year'] <= max_year)]

def filter_by_states(df, states):
    return df[df['state'].isin(states)]