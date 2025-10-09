from dash import Dash, Input, Output, html, dcc
import plotly.express as px
import pandas as pd
from data import load_debt_data, total_annual_debt_by_states, filter_by_year
import json

app = Dash()

df = load_debt_data()

# sample data for each state
state_data = pd.DataFrame({
    'state': df['state'].unique(),
    'value': [1] * len(df['state'].unique())
})
with open("data/germany.geojson", "r") as f:
    germany_geojson = json.load(f)

germany_map = px.choropleth(state_data, geojson=germany_geojson,
                    locations="state", featureidkey="properties.NAME_1",
                    projection="mercator"
                   )
germany_map.update_geos(fitbounds="locations", visible=False)
germany_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})


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
        dcc.Slider(
            id='time-slider',
            min=df['time'].min(),
            max=df['time'].max(),
            step=1,
            value=df['time'].min(),
            marks={year: str(year) for year in range(df['time'].min(), df['time'].max() + 1)}
        )
    ]),
])

@app.callback(
    Output("line-chart-1", "figure"),
    Output("line-chart-2", "figure"),
    Input("state-dropdown", "value"))
def update_line_charts(selected_states):
    state_debt = total_annual_debt_by_states(df, selected_states)
    line_chart = px.line(
        state_debt,
        x='time',
        y='total_annual_debt',
        color='state',
        title='Total Annual Debt by State Over Time',
        labels={
            'time': 'Year',
            'total_annual_debt': 'Total Annual Debt (Bn. EUR)',
            'state': 'State'
        }
    )
    return line_chart, line_chart

@app.callback(
    Output("bar-chart-1", "figure"),
    Output("bar-chart-2", "figure"),
    Input("time-slider", "value"),
    Input("state-dropdown", "value"))
def update_bar_charts(year, selected_states):
    state_debt = total_annual_debt_by_states(df, selected_states)
    state_debt_year = filter_by_year(state_debt, year)
    bar_chart = px.bar(
        state_debt_year,
        x='state',
        y='total_annual_debt',
        title=f'Total Annual Debt by State in {year}',
        labels={
            'state': 'State',
            'total_annual_debt': 'Total Annual Debt (Bn. EUR)'
        }
    )
    return bar_chart, bar_chart


if __name__ == '__main__':
    app.run(debug=True)