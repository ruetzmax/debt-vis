import pandas as pd

def load_debt_data():
    df = pd.read_csv('data/debt_92-05.csv', sep=';')
    df2 = pd.read_csv('data/debt_06-09.csv', sep=';')
    df = pd.concat([df, df2], ignore_index=True)
    df3 = pd.read_csv('data/debt_10-24.csv', sep=';')
    df = pd.concat([df, df3], ignore_index=True)
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
    result = merged[['state', 'year', 'debt_per_person_eur']].rename(columns={'debt_per_person_eur': 'value'})
    result.attrs['unit'] = 'EUR per capita'
    return result

def normalized_unemployment_per_capita():
    unemp_df = total_annual_unemployment()
    pop_long = get_population_long()
    unemp_df['year'] = unemp_df['year'].astype(int)
    pop_long['year'] = pop_long['year'].astype(int)
    merged = unemp_df.merge(pop_long, on=['state', 'year'], how='inner')
    merged['unemployment_rate_percent'] = (merged['value'] / merged['population'] * 100).round(2)
    result = merged[['state', 'year', 'unemployment_rate_percent']].rename(columns={'unemployment_rate_percent': 'value'})
    result.attrs['unit'] = '% of population'
    return result

def load_graduation_rates():
    df = pd.read_csv('data/graduation_rates_per_state.csv', sep=';')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df[df['sex'] == 'Total']
    df_long = df.melt(id_vars=['state'], var_name='year', value_name='value')
    df_long['year'] = pd.to_numeric(df_long['year'], errors='coerce')
    df_long = df_long.dropna(subset=['year'])
    df_long['year'] = df_long['year'].astype(int)
    df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce')
    df_long.attrs['unit'] = '% graduation rate'
    return df_long

def load_recipients_of_benefits():
    df = pd.read_csv('data/recipients_of_benefits.csv', sep=';', on_bad_lines='skip')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df[df['sex'] == 'Total']
    year_cols = [c for c in df.columns if str(c).strip().replace('.', '').isdigit()]
    state_cols = [c for c in df.columns if c not in ['sex'] + year_cols]
    state_col = state_cols[0]
    df_long = df.melt(id_vars=[state_col], value_vars=year_cols,
                      var_name='year', value_name='value')
    df_long = df_long.rename(columns={state_col: 'state'})
    df_long['year'] = pd.to_numeric(df_long['year'], errors='coerce').astype(int)
    df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce')
    result = df_long.fillna(0)
    result.attrs['unit'] = 'recipients count'
    result = result.groupby(['state','year'], as_index=False).agg({'value': 'sum'})
    return result

def get_dataset_unit(dataset_name, features_dict):
    """Helper function to get the unit for a dataset"""
    if dataset_name in features_dict:
        return getattr(features_dict[dataset_name], 'attrs', {}).get('unit', 'units')
    return 'units'

def load_expenditure_on_public_schools():
    df = pd.read_csv('data/expenditure_in_public_school_per_pupil.csv', sep=';')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.rename(columns={'1_variable_attribute_label': 'state'})
    df['time'] = pd.to_datetime(df['time']).dt.year
    df = df.rename(columns={'time': 'year'})
    df.attrs['unit'] = 'EUR per pupil'
    return df.dropna()

def combine_features(feature_dict, chosen_features):
    '''
    Currently it is set up to make a new frame depending on chosen features since 
    not all features have the same years available.
    '''
    
    # Select relevant features
    # Always load debt
    debt = normalized_debt_per_capita()[['state', 'year', 'value']]
    debt_grouped = debt.groupby(['state','year'], as_index=False).agg({'value': 'sum'})
    combined = debt_grouped.sort_values(['state', 'year'])
    combined = combined.rename(columns={'value': 'Debt'})

    # Load other dataframes
    feature_frames = [feature_dict[feature] for feature in chosen_features]

    # Find the time interval available for all features
    min_year = max([min(df['year']) for df in feature_frames])
    max_year = min([max(df['year']) for df in feature_frames])

    # Filter all features to this interval
    combined = filter_by_years(combined, min_year, max_year)
    feature_frames = [filter_by_years(df, min_year, max_year) for df in feature_frames]

    # Put all value columns into combined frame
    for idx, feature in enumerate(chosen_features):
        combined[feature] = feature_frames[idx].sort_values(['state', 'year'])['value'].values

    return combined