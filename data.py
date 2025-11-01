import pandas as pd


#TODO maybe a wrapper function to load different dataframes in case we extend to a lot of features?
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

def combine_vars(chosen_vars=None):
    '''
    TODO
    - load various variable dfs
    - group and sort
    - select based on min and max year present
    - put variables sorted into one frame
    '''
    # Select relevant features (currently just these two for testing)
    debt = load_debt_data()[['state', 'year', 'value']]
    unemployment = total_annual_unemployment()

    debt_grouped = debt.groupby(['state','year'], as_index=False).agg({'value': 'sum'})

    # Debt data covers smaller span so using that now, extend to more features
    min_year = min(debt['year'])
    max_year = max(debt['year'])

    unemployment = filter_by_years(unemployment, min_year, max_year)

    combined = debt_grouped.sort_values('state')
    combined['unemployment'] = unemployment.sort_values('state')['value'].values
    combined = combined.rename(columns={'value': 'debt'})

    return combined
def load_population_density():
    df = pd.read_csv('data/population_density_95-23.csv', sep=';')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df

AREA_KM2 = {
	'Baden-Württemberg': 35752,
	'Bayern': 70549,
	'Berlin': 892,
	'Brandenburg': 29477,
	'Bremen': 404,
	'Hamburg': 755,
	'Hessen': 21115,
	'Mecklenburg-Vorpommern': 23173,
	'Niedersachsen': 47618,
	'Nordrhein-Westfalen': 34083,
	'Rheinland-Pfalz': 19847,
	'Saarland': 2569,
	'Sachsen': 18413,
	'Sachsen-Anhalt': 20445,
	'Schleswig-Holstein': 15763,
	'Thüringen': 16172,
}

def population_from_density():
    df = load_population_density()
    df_population = df.copy()
    for state in AREA_KM2:
        if state in df.columns:
            density = pd.to_numeric(df[state])
            population = (density * AREA_KM2[state]).round().astype('Int64')  
            df_population[state] = population
    return df_population

def get_population_long():
    df = population_from_density()
    df['year'] = pd.to_datetime(df['year'], errors='coerce').dt.year.astype('Int64')
    df_long = df.melt(id_vars=['year'], var_name='state', value_name='population')
    return df_long.dropna(subset=['year']) 

def normalized_debt_per_capita():
    debt_df = total_annual_debt()
    pop_long = get_population_long()
    debt_df['year'] = debt_df['year'].astype(int)
    pop_long['year'] = pop_long['year'].astype(int)
    merged = debt_df.merge(pop_long, on=['state', 'year'], how='inner')
    merged['debt_per_person_eur'] = (merged['value'] * 1_000_000 / merged['population']).round(2)
    return merged[['state', 'year', 'debt_per_person_eur']].rename(columns={'debt_per_person_eur': 'value'})

def normalized_unemployment_per_capita():
    unemp_df = total_annual_unemployment()
    pop_long = get_population_long()
    unemp_df['year'] = unemp_df['year'].astype(int)
    pop_long['year'] = pop_long['year'].astype(int)
    merged = unemp_df.merge(pop_long, on=['state', 'year'], how='inner')
    merged['unemployment_rate_percent'] = (merged['value'] / merged['population'] * 100).round(2)
    return merged[['state', 'year', 'unemployment_rate_percent']].rename(columns={'unemployment_rate_percent': 'value'})
