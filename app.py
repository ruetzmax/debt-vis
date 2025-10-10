from dash import Dash, Input, Output, html, dcc
import plotly.express as px
import pandas as pd
from data import load_debt_data, total_annual_debt, total_annual_unemployment, filter_by_year, filter_by_years, filter_by_states
import json

app = Dash()

features = {
    "Debt": total_annual_debt(),
    "Unemployment": total_annual_unemployment()
}

# MAP
df = features["Debt"]
state_data = pd.DataFrame({
    'state': df['state'],
    'value': df['value']
})
with open("data/germany.geojson", "r") as f:
    germany_geojson = json.load(f)

germany_map = px.choropleth(
                    state_data, 
                    geojson=germany_geojson,
                    locations="state", 
                    featureidkey="properties.NAME_1", 
                    color="value",
                    projection="mercator"
                   )
germany_map.update_geos(fitbounds="locations", visible=False)
germany_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

# TIME SLIDER
min_year = max(feature_df['year'].min() for feature_df in features.values())
max_year = min(feature_df['year'].max() for feature_df in features.values())
time_slider = dcc.Slider(
            id='time-slider',
            min=min_year,
            max=max_year,
            step=1,
            value=min_year,
            marks={year: str(year) for year in range(min_year, max_year + 1)}
        )

def get_bar_chart(title, data, year, selected_states):
    filtered_data = filter_by_year(data, year)
    filtered_data = filter_by_states(filtered_data, selected_states)

    bar_chart = px.bar(
        filtered_data,
        x='state',
        y='value',
        title=f'{title} in {year}',
        labels={
            'state': 'State',
            'value': title
        }
    )
    return bar_chart

def get_line_chart(title, data, selected_states, min_year, max_year):
    filtered_data = filter_by_states(data, selected_states)
    filtered_data = filter_by_years(filtered_data, min_year, max_year)
    
    line_chart = px.line(
        filtered_data,
        x='year',
        y='value',
        color='state',
        title=f'{title} Over Time',
        labels={
            'year': 'Year',
            'value': title,
            'state': 'State'
        }
    )
    return line_chart

# LAYOUT
app.layout = html.Div(children=[
    html.H1(children='German Debt and Socioeconomic Factors'),
    html.Div(
        id='content',
        children=[
            html.Div(
                id='map-container',
                children=[
                    dcc.Graph(figure=germany_map)
                ]
            ),
            html.Div(
                id='filters-container',
                children=[
                    html.Label("Choose States"),
                    dcc.Dropdown(
                        df['state'].unique(),
                        ['Berlin'],
                        multi=True,
                        id="state-dropdown"
                    ),
                    html.Label("Features"),
                    dcc.Checklist(
                        list(features.keys()),
                        [list(features.keys())[0]],
                        id="feature-checklist"
                    )
                ]
            ),
            html.Div(
                id='charts-container',
                children=[
                    html.Div(
                        id='line-charts-container', children=
                        [
                            dcc.Graph(id="line-chart-1"),
                            dcc.Graph(id="line-chart-2")
                        ]
                    ),
                    html.Div(
                        id='bar-charts-container', children=[
                            dcc.Graph(id="bar-chart-1"),
                            dcc.Graph(id="bar-chart-2")
                        ]
                    )
                ]
            )
        ]
    ),

    html.Div(id='time-slider-container', children=[
        time_slider
    ]),
])

@app.callback(
    Output("line-charts-container", "children"),
    Input("state-dropdown", "value"),
    Input("feature-checklist", "value"),
    Input("time-slider", "min"),
    Input("time-slider", "max"))
def update_line_charts(selected_states, selected_features, min_year, max_year):
    charts = []
    for title, data in features.items():
        if title not in selected_features:
            continue
        chart = get_line_chart(title, data, selected_states, min_year, max_year)
        charts.append(dcc.Graph(figure=chart))
    return charts

@app.callback(
    Output("bar-charts-container", "children"),
    Input("time-slider", "value"),
    Input("state-dropdown", "value"),
    Input("feature-checklist", "value"))
def update_bar_charts(year, selected_states, selected_features):
    charts = []
    for title, data in features.items():
        if title not in selected_features:
            continue
        chart = get_bar_chart(title, data, year, selected_states)
        charts.append(dcc.Graph(figure=chart))
    return charts

@app.callback(
    Output("time-slider", "min"),
    Output("time-slider", "max"),
    Output("time-slider", "value"),
    Output("time-slider", "marks"),
    Input("feature-checklist", "value"),
    Input("time-slider", "value"))
def update_time_slider(selected_features, current_year):
    if not selected_features:
        return 2000, 2020, 2000, {year: str(year) for year in range(2000, 2021)}
    selected_dfs = [features[feature] for feature in selected_features if feature in features]
    if not selected_dfs:
        return 2000, 2020, 2000, {year: str(year) for year in range(2000, 2021)}
    min_year = max(df['year'].min() for df in selected_dfs)
    max_year = min(df['year'].max() for df in selected_dfs)
    marks = {year: str(year) for year in range(min_year, max_year + 1)}
    value = current_year if min_year <= current_year <= max_year else min_year
    return min_year, max_year, value, marks

if __name__ == '__main__':
    app.run(debug=True)