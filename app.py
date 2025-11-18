from dash import Dash, Input, Output, State, html, dcc, ctx, no_update, ALL
import plotly.express as px
import pandas as pd
from data import load_debt_data, total_annual_debt, total_annual_unemployment, filter_by_year, filter_by_years, filter_by_states, population_from_density, normalized_debt_per_capita, normalized_unemployment_per_capita, load_recipients_of_benefits, load_graduation_rates, get_dataset_unit, load_expenditure_on_public_schools
import json
import math
import json as json_lib


app = Dash()

features = {
    "Debt": normalized_debt_per_capita(),
    "Unemployment": normalized_unemployment_per_capita(),
    "Graduation Rates": load_graduation_rates(),
    "Recipients of Benefits": load_recipients_of_benefits(),
    "Expenditure on Public Schools": load_expenditure_on_public_schools()
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
                    projection="mercator",
                    title="Germany Economic Indicators Map"
                   )
germany_map.update_geos(fitbounds="locations", visible=False)
germany_map.update_layout(
    margin={"r":0,"t":40,"l":0,"b":0}, 
    title={
        "text": "Debt by State",
        "y": 0.98,
        "x": 0.5,
        "xanchor": "center",
        "yanchor": "top",
        "font": {"size": 16}
    }
)

#Secondary map - variable features
secondary_df = features["Unemployment"]
secondary_data = pd.DataFrame({
    'state': secondary_df['state'],
    'value': secondary_df['value']
})
with open("data/germany.geojson", "r") as f:
    germany_geojson = json.load(f)

secondary_map = px.choropleth(
                    secondary_data, 
                    geojson=germany_geojson,
                    locations="state", 
                    featureidkey="properties.NAME_1", 
                    color="value",
                    color_continuous_scale="delta",
                    projection="mercator",
                    title="Germany Economic Indicators Map"
                   )
secondary_map.update_geos(fitbounds="locations", visible=False)
secondary_map.update_layout(
    margin={"r":0,"t":40,"l":0,"b":0}, 
    title={
        "text": "*Feature* by State",
        "y": 0.98,
        "x": 0.5,
        "xanchor": "center",
        "yanchor": "top",
        "font": {"size": 16}
    }
)

#Difference map - debt vs selected feature
debt_df = features["Debt"]
other_df = features["Unemployment"]


debt_scaled = (debt_df["value"] - debt_df["value"].min()) / (debt_df["value"].max() - debt_df["value"].min())
other_scaled = (other_df["value"] - other_df["value"].min()) / (other_df["value"].max() - other_df["value"].min())

df_difference = debt_df.copy()
#Consider if this should be done absolute or other order?
df_difference["value"] = debt_scaled - other_scaled
mapdata_difference = pd.DataFrame({
    'state': df_difference['state'],
    'value': df_difference['value']
})
difference_map = px.choropleth(
                    mapdata_difference, 
                    geojson=germany_geojson,
                    locations="state", 
                    featureidkey="properties.NAME_1", 
                    color="value",
                    projection="mercator",
                    title="Difference Map"
                   )
difference_map.update_geos(fitbounds="locations", visible=False)
difference_map.update_layout(
    margin={"r":0,"t":40,"l":0,"b":0}, 
    title={
        "text": "Difference in debt vs *attribute*",
        "y": 0.98,
        "x": 0.5,
        "xanchor": "center",
        "yanchor": "top",
        "font": {"size": 16}
    }
)

# TIME SLIDER - calculate intersection of all features at startup
min_years = []
max_years = []

for feature_df in features.values():
    if 'year' in feature_df.columns:
        years = feature_df['year'].dropna()
        if len(years) > 0:
            min_years.append(years.min())
            max_years.append(years.max())

if min_years and max_years:
    min_year = max(min_years)  # Latest start year (intersection)
    max_year = min(max_years)  # Earliest end year (intersection)
else:
    min_year, max_year = 2000, 2020

time_slider = dcc.Slider(
            id='time-slider',
            min=min_year,
            max=max_year,
            step=1,
            value=min_year,
            marks={year: str(year) for year in range(min_year, max_year + 1)}
        )

range_slider = dcc.RangeSlider(
            id='time-range-slider',
            min=min_year,
            max=max_year,
            step=1,
            value=[min_year, max_year],
            marks={year: str(year) for year in range(min_year, max_year + 1)}
        )

secondary_features = features.copy()
del secondary_features["Debt"]

# LAYOUT
app.layout = html.Div(children=[
    html.H1(children='German Debt and Socioeconomic Factors'),
    html.Div(
        id='main-layout',
        style={
            'display': 'flex',
            'height': '80vh',
            'gap': '20px'
        },
        children=[
            html.Div(
                id='left-side',
                style={
                    'flex': '1',
                    'display': 'flex',
                    'flex-direction': 'column',
                    'gap': '20px'
                },
                children=[
                    html.Div(
                        id='timewheel-of-death',
                        style={
                            'flex': '4',
                            'background-color': 'white',
                            'border-radius': '8px',
                            'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
                            'padding': '20px',
                            'display': 'flex',
                            'align-items': 'center',
                            'justify-content': 'center',
                            'color': '#666',
                            'font-size': '18px'
                        },
                        children=[
                            html.Div("Timewheel of Death", style={'text-align': 'center'})
                        ]
                    ),
                    html.Div(
                        id='controls-container',
                        style={
                            'flex': '1',
                            'background-color': 'white',
                            'border-radius': '8px',
                            'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
                            'padding': '15px',
                            'display': 'flex',
                            'flex-direction': 'column',
                            'gap': '10px'
                        },
                        children=[
                            html.Div(
                                id='filters-section',
                                children=[
                                    html.Label("Choose States", style={'font-weight': 'bold', 'margin-bottom': '3px', 'display': 'block', 'font-size': '12px'}),
                                    dcc.Dropdown(
                                        df['state'].unique(),
                                        ['Berlin'],
                                        multi=True,
                                        id="state-dropdown",
                                        style={'margin-bottom': '8px', 'font-size': '12px'}
                                    ),
                                    html.Div(
                                        style={'display': 'flex'},
                                        children=[
                                            html.Div(
                                                style={'flex' : '1', 'display' : 'block'}, 
                                                children=[
                                                    html.Label("Features", style={'font-weight': 'bold', 'margin-bottom': '3px', 'display': 'block', 'font-size': '12px'}),
                                                    dcc.Checklist(
                                                        list(features.keys()),
                                                        [list(features.keys())[0]],
                                                        id="feature-checklist",
                                                        style={'font-size': '11px'}
                                                    )
                                                ]
                                            ),
                                            html.Div(
                                                style={'flex' : '1', 'display' : 'block'}, 
                                                children=[
                                                    html.Label("Secondary map feature", style={'font-weight': 'bold', 'margin-bottom': '3px', 'display': 'block', 'font-size': '12px'}),
                                                    dcc.Dropdown(
                                                        list(secondary_features.keys()),
                                                        list(secondary_features.keys())[0],
                                                        id="secondary-feature-dropdown",
                                                        style={'font-size': '11px'}
                                                    )
                                                ]
                                            )
                                        ]
                                    )
                                ]
                            ),
                            html.Div(
                                id='time-controls',
                                children=[
                                    html.Label("Time Selection", style={'font-weight': 'bold', 'margin-bottom': '5px', 'display': 'block', 'font-size': '12px'}),
                                    html.Div(
                                        style={'display': 'flex', 'align-items': 'center', 'gap': '10px'},
                                        children=[
                                            html.Div(id='single-slider-div', children=[time_slider], style={'flex': '1'}),
                                            html.Div(id='range-slider-div', children=[range_slider], style={'flex': '1', 'display': 'none'}),
                                            html.Button("Switch to Interval", id='time-slider-switch-button', 
                                                      style={'flex': '0 0 auto', 'height': '30px', 'font-size': '10px', 'padding': '5px 8px'})
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            ),
            html.Div(
                id='right-side',
                style={
                    'flex': '1',
                    'background-color': 'white',
                    'border-radius': '8px',
                    'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
                    'padding': '0',
                    'display': 'flex',
                    'flex-direction': 'column'
                },
                children=[
                    dcc.Graph(id='difference-map', figure=difference_map, style={'flex': '2', 'height': '100%', 'padding' : '5px'}),
                    html.Div(style={
                        'flex' : '1',
                        'display' : 'flex',
                        'flex-direction' : 'row',
                    }, children=[
                        dcc.Graph(id='debt-map', figure=germany_map, style={'flex': '1', 'height': '100%', 'padding' : '5px'}),
                        dcc.Graph(id='secondary-map', figure=secondary_map, style={'flex': '1', 'height': '100%', 'padding' : '5px'})
                ])]
            )
        ]
    ),
    dcc.Store(id='time-slider-mode-store', data='single'),
])

#Added clickData outputs to enable clicking same state twice
@app.callback(
    Output("state-dropdown", "value"),
    Output("difference-map", "clickData"),
    Output("debt-map", "clickData"),
    Output("secondary-map", "clickData"),
    Input("difference-map", "clickData"),
    Input("debt-map", "clickData"),
    Input("secondary-map", "clickData"),
    State("state-dropdown", "value")
)
def update_state_selection(clickData1, clickData2, clickData3, current_selection):
    if not ctx.triggered:
        return current_selection, None, None, None
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == "difference-map":
        clickData = clickData1
    elif trigger_id == "debt-map":
        clickData = clickData2
    else:
        clickData = clickData3
    
    if clickData is None:
        return current_selection, None, None, None
    
    state_clicked = clickData["points"][0]["location"]
    selected = current_selection.copy() if current_selection else []
    if state_clicked in selected:
        selected.remove(state_clicked)
    else:
        selected.append(state_clicked)
        
    return selected, None, None, None

#Input("feature-checklist", "value"),
@app.callback(
    Output("debt-map", "figure"),
    Input("time-slider", "value"),
    Input("time-slider", "min"),
    Input("time-slider", "max"),
    Input("time-range-slider", "value"),
    Input("time-slider-mode-store", "data"),
)
def update_map(single_value, single_min, single_max, range_value, slider_mode):
    map_feature = "Debt"
    data_df = features.get("Debt")

    unit = get_dataset_unit(map_feature, features)

    if slider_mode == "single":
        year = single_value if single_value is not None else single_min
        filtered = filter_by_year(data_df, year)
        title = f"{map_feature} in {year} ({unit})"
    else:
        start, end = (range_value if range_value and len(range_value) == 2 else (single_min, single_max))
        filtered = filter_by_years(data_df, start, end)
        filtered = filtered.groupby("state", as_index=False)["value"].mean()
        title = f"{map_feature} averaged {start} - {end} ({unit})"

    if filtered.empty:
        state_data = pd.DataFrame({
            "state": ["Berlin"],
            "value": [0]
        })
        title = f"No data available for {map_feature}"
    else:
        state_data = pd.DataFrame({
            "state": filtered["state"],
            "value": filtered["value"]
        })

    fig = px.choropleth(
        state_data,
        geojson=germany_geojson,
        locations="state",
        featureidkey="properties.NAME_1",
        color="value",
        projection="mercator",
        title="Germany Economic Indicators Map"
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0},
        title={
            "text": title,
            "y": 0.98,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 16}
        }
    )
    return fig


@app.callback(
    Output('single-slider-div', 'style'),
    Output('range-slider-div', 'style'),
    Output('time-slider-mode-store', 'data'),
    Output('time-slider-switch-button', 'children'),
    Input('time-slider-switch-button', 'n_clicks'),
    State('time-slider-mode-store', 'data')
)
def switch_time_slider_mode(n_clicks, current_mode):
    current_mode = current_mode or 'single'
    if current_mode == 'single':
        single_style = {'display': 'none'}
        range_style = {'display': 'block'}
        new_mode = 'range'
        button_text = "Switch to Single"
    else:
        single_style = {'display': 'block'}
        range_style = {'display': 'none'}
        new_mode = 'single'
        button_text = "Switch to Interval"
        
    return single_style, range_style, new_mode, button_text

@app.callback(
    Output("time-slider", "min"),
    Output("time-slider", "max"),
    Output("time-slider", "value"),
    Output("time-slider", "marks"),
    Output("time-range-slider", "min"),
    Output("time-range-slider", "max"),
    Output("time-range-slider", "value"),
    Output("time-range-slider", "marks"),
    Input("feature-checklist", "value"),
    State("time-slider", "value"),
    State("time-range-slider", "value"))
def update_time_slider(selected_features, current_single_year, current_range_value):
    if not selected_features:
        default_marks = {year: str(year) for year in range(2000, 2021)}
        return 2000, 2020, 2000, default_marks, 2000, 2020, [2000, 2020], default_marks
    
    selected_dfs = [features[feature] for feature in selected_features if feature in features]
    if not selected_dfs:
        default_marks = {year: str(year) for year in range(2000, 2021)}
        return 2000, 2020, 2000, default_marks, 2000, 2020, [2000, 2020], default_marks
    
    if len(selected_features) == 1:
        df = selected_dfs[0]
        min_year = int(df['year'].min())
        max_year = int(df['year'].max())
    else:
        min_years = [int(df['year'].min()) for df in selected_dfs]
        max_years = [int(df['year'].max()) for df in selected_dfs]
        min_year = max(min_years)
        max_year = min(max_years)
        
        if min_year > max_year:
            default_marks = {year: str(year) for year in range(2000, 2021)}
            return 2000, 2020, 2000, default_marks, 2000, 2020, [2000, 2020], default_marks
    
    marks = {year: str(year) for year in range(min_year, max_year + 1)}
    
    single_value = current_single_year if min_year <= current_single_year <= max_year else min_year
    
    if current_range_value and len(current_range_value) == 2:
        range_start = max(min_year, min(current_range_value[0], max_year))
        range_end = min(max_year, max(current_range_value[1], min_year))
        range_value = [range_start, range_end]
    else:
        range_value = [min_year, max_year]
    
    return (min_year, max_year, single_value, marks, 
            min_year, max_year, range_value, marks)

if __name__ == '__main__':
    app.run(debug=True)