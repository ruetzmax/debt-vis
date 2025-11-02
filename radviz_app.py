from dash import Dash, Input, Output, State, html, dcc, ctx, ALL
import plotly.express as px
from radviz_plotly import RadViz2D
import pandas as pd
from data import load_debt_data, total_annual_debt, total_annual_unemployment, filter_by_year, filter_by_years, filter_by_states, population_from_density, normalized_debt_per_capita, normalized_unemployment_per_capita, combine_vars
from radviz import get_radviz_coords, radviz_from_mpl
import json
import math
import json as json_lib


app = Dash()

features = {
    "Debt": normalized_debt_per_capita(),
    "Unemployment": normalized_unemployment_per_capita(),
    "Dummy": total_annual_unemployment()
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

range_slider = dcc.RangeSlider(
            id='time-range-slider',
            min=min_year,
            max=max_year,
            step=1,
            value=[min_year, max_year],
            marks={year: str(year) for year in range(min_year, max_year + 1)}
        )

def get_bar_chart(title, data, min_year, max_year, current_year, selected_states):
    filtered_data = filter_by_states(data, selected_states)
    
    if current_year:
        filtered_data = filter_by_year(filtered_data, current_year)
        graph_title = f'{title} in {current_year}'
    else:
        filtered_data = filter_by_years(filtered_data, min_year, max_year)
        filtered_data = (
            filtered_data.groupby('state', as_index=False)['value']
            .mean()
        )
        graph_title = f'{title} averaged {min_year} - {max_year}'


    bar_chart = px.bar(
        filtered_data,
        x='state',
        y='value',
        title=graph_title,
        labels={
            'state': 'State',
            'value': title
        }
    )
    
    bar_chart.update_layout(
        title={
            'text': graph_title,
            'font': {'size': 16, 'family': 'Arial, sans-serif', 'weight': 'bold'},
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    return bar_chart

def get_line_chart(title, data, selected_states, min_year, max_year, current_year=None):
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
        },
        markers=True  
    )
    
    line_chart.update_layout(
        title={
            'text': f'{title} Over Time',
            'font': {'size': 16, 'family': 'Arial, sans-serif', 'weight': 'bold'},
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    
    if current_year is not None:
        line_chart.add_vline(
            x=current_year,
            line_width=2,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Current: {current_year}",
            annotation_position="top right"
        )
    
    return line_chart

def get_radviz(selected_features, selected_states, min_year, max_year, current_year=None):
    # for now selected features does nothing since combine_vars is still hardcoded
    data = combine_vars()
    selected_features = ['debt', 'unemployment']

    # # Add encoded state for color
    # data['state_enc']=data['state'].astype('category').cat.codes

    filtered_data = filter_by_states(data, selected_states)
    filtered_data = filter_by_years(filtered_data, min_year, max_year)

    radviz_chart = radviz_from_mpl(data, min_year, max_year)

    return radviz_chart
    

# LAYOUT
app.layout = html.Div(children=[
    html.H1(children='German Debt and Socioeconomic Factors'),
    html.Div(
        id='content',
        children=[
            html.Div(
                id='map-container',
                children=[
                    dcc.Graph(id='debt-map', figure=germany_map, className='dash-graph', style={"height": "600px"})
                ],
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
                id = 'charts-panel',
                style={"display": "flex", "flexDirection": "column", "height": "100%", "minHeight": "600px"},
                children=[
                    html.Div(
                        id='overview-section',
                        style={"flex": "1", "minHeight": 0, "overflow": "hidden"},
                        className='chart-section',
                        children=[
                            # PCS chart
                            html.Div(id="overview-container", className="chart-container"),
                        ]
                    ),
                    html.Div(
                        id='chart-section',
                        style={"flex": "1", "minHeight": 0, "overflow": "hidden"},
                        className='chart-section',
                        children=[
                            html.Div(className='chart-header', children=[
                                html.H3("Charts", className='chart-title'),
                                html.Button("Switch to Bar Charts", id="graph-type-switch-button"),
                                html.Div(className='chart-navigation', children=[
                                    html.Button("◀", id="prev-chart", className="nav-button"),
                                    html.Span(id="chart-page-indicator", children="1/1"),
                                    html.Button("▶", id="next-chart", className="nav-button"),
                                ])
                            ]),
                            # Line & bar charts
                            html.Div(id="charts-container", className="chart-container")
                        ]
                    ),
                ]
            )
        ]
    ),

    html.Div(id='time-slider-container', children=[
        html.Div(id='single-slider-div', children=[time_slider]),
        html.Div(id='range-slider-div', children=[range_slider], style={'display': 'none'}),
        html.Button("Switch to Interval", id='time-slider-switch-button')
    ]),
    
    
    # Data stores
    dcc.Store(id='chart-order-store', data=[i for i, key in enumerate(features.keys()) if key in [list(features.keys())[0]]]),
    dcc.Store(id='graph-type-store', data='line'),
    dcc.Store(id='time-slider-mode-store', data='single'),
    
])

@app.callback(
    Output("state-dropdown", "value"),
    Input("debt-map", "clickData"),
    State("state-dropdown", "value")
)
def update_state_selection(clickData, current_selection):
    if clickData is None:
        return current_selection
    
    state_clicked = clickData["points"][0]["location"]
    selected = current_selection.copy() if current_selection else []
    if state_clicked in selected:
        selected.remove(state_clicked)
    else:
        selected.append(state_clicked)
        
    return selected

@app.callback(
    Output("debt-map", "figure"),
    Input("feature-checklist", "value"),
    Input("time-slider", "value"),
    Input("time-slider", "min"),
    Input("time-slider", "max"),
    Input("time-range-slider", "value"),
    Input("time-slider-mode-store", "data"),
)
def update_map(selected_features, single_value, single_min, single_max, range_value, slider_mode):
    data_df = features.get("Debt")

    # determine time selection
    if slider_mode == "single":
        year = single_value if single_value is not None else single_min
        filtered = filter_by_year(data_df, year)
        title = f"Debt in {year}"
    else:
        start, end = (range_value if range_value and len(range_value) == 2 else (single_min, single_max))
        filtered = filter_by_years(data_df, start, end)
        filtered = filtered.groupby("state", as_index=False)["value"].mean()
        title = f"Debt averaged {start} - {end}"

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


chart_page = 0
@app.callback(
    [Output("charts-container", "children"),
     Output("chart-page-indicator", "children")],
    [Input("state-dropdown", "value"),
     Input("feature-checklist", "value"),
     Input("time-slider", "min"),
     Input("time-slider", "max"),
     Input("time-slider", "value"),
     Input("time-range-slider", "min"),
     Input("time-range-slider", "max"),
     Input("time-range-slider", "value"),
     Input("prev-chart", "n_clicks"),
     Input("next-chart", "n_clicks"),
     Input("chart-order-store", "data"),
     Input("graph-type-store", "data"),
     Input("time-slider-mode-store", "data")],
    
    
)
def update_charts(selected_states, selected_features, single_min, single_max, single_value, range_min, range_max, range_value, prev_clicks, next_clicks, chart_order, graph_type, slider_mode):
    # Get the callback context
    triggered = ctx.triggered_id
    all_charts = []

    
    if slider_mode == 'single':
        min_year = single_min
        max_year = single_max
        current_year = single_value
    else:
        min_year = range_value[0] if range_value else range_min
        max_year = range_value[1] if range_value else range_max
        current_year = None
    
    # Create all charts first, ordered by chart_order
    features_list = list(features.keys())
    
    for chart_index in (chart_order or []):
        if chart_index >= len(features_list):
            continue
        
        title = features_list[chart_index]
        if title not in selected_features:
            continue
            
        data = features[title]
        
        if graph_type == "line":
            chart = get_line_chart(title, data, selected_states, min_year, max_year, current_year)
        else:
            chart = get_bar_chart(title, data, min_year, max_year, current_year, selected_states)
            
        chart_div = html.Div(children=[
            dcc.Graph(figure=chart, style={"height": "100%", "width": "100%"}),
            dcc.Dropdown(
                id={"type": "switch-dropdown", "index": chart_index},
                options=[{"label": "Switch feature...", "value": "switch"}] + [{"label": feature, "value": feature} for feature in selected_features],
                value="switch",
                clearable=False,
            )
        ], style={"flex": "1", "minHeight":0, "overflow": "hidden"})
        all_charts.append(chart_div)

    # Handle pagination
    total_charts = len(all_charts)
    if total_charts == 0:
        return [html.Div("No charts to display")], "0/0"
    
    charts_per_page = 2
    total_pages = max(1, math.ceil(total_charts / charts_per_page))
    
    global chart_page
    
    if triggered == 'prev-chart':
        chart_page = (chart_page - 1) % total_pages
    elif triggered == 'next-chart':
        chart_page = (chart_page + 1) % total_pages
    elif triggered == 'feature-checklist':
        # Reset to first page when features change
        chart_page = 0
        
    chart_page = max(0, min(chart_page, total_pages - 1))
    
    start_idx = chart_page * charts_per_page
    end_idx = min(start_idx + charts_per_page, total_charts)
    current_charts = all_charts[start_idx:end_idx]
    
    # Update page indicator
    page_indicator = f"{chart_page + 1}/{total_pages}"
    
    return current_charts, page_indicator

@app.callback(
    Output("overview-container", "children"),
    [Input("state-dropdown", "value"),
     Input("feature-checklist", "value"),
     Input("time-slider", "min"),
     Input("time-slider", "max"),
     Input("time-slider", "value"),
     Input("time-range-slider", "min"),
     Input("time-range-slider", "max"),
     Input("time-range-slider", "value"),
     Input("time-slider-mode-store", "data")]
)
def update_radviz(selected_states, selected_features, single_min, single_max, single_value, range_min, range_max, range_value, slider_mode):


    if slider_mode == 'single':
        min_year = single_min
        max_year = single_max
        current_year = single_value
    else:
        min_year = range_value[0] if range_value else range_min
        max_year = range_value[1] if range_value else range_max
        current_year = None

    radviz_chart = get_radviz(selected_features, selected_states, min_year, max_year, current_year)

    radviz_div = html.Div(children=[
            dcc.Graph(figure=radviz_chart, style={"height": "100%", "width": "100%"}),
        ], style={"flex": "1", "minHeight": 0, "overflow": "hidden"})
    
    return radviz_div

@app.callback(
    Output("graph-type-store", "data"),
    Output("graph-type-switch-button", "children"),
    Input("graph-type-switch-button", "n_clicks"),
    State("graph-type-store", "data"),
    prevent_initial_call=True
)
def switch_graph_type(n_clicks, current_type):
    current_type = current_type or "line"
    new_type = "bar" if current_type == "line" else "line"
    button_text = "Switch to Line Charts" if new_type == "bar" else "Switch to Bar Charts"
    return new_type, button_text

@app.callback(
    Output('chart-order-store', 'data'),
    Input({"type": "switch-dropdown", "index": ALL}, "value"),
    Input("feature-checklist", "value"),
    State('chart-order-store', 'data'),
    prevent_initial_call=False
)
def switch_feature_order(dropdown_values, selected_features, chart_order):  
    # if a selected feature is not in the chart order, add it to the end
    for feature in selected_features or []:
        if feature not in features:
            continue
        feature_index = list(features.keys()).index(feature)
        if feature_index not in chart_order:
            chart_order.append(feature_index)
            
    # if a feature in chart order is not selected, remove it
    chart_order = [idx for idx in chart_order if list(features.keys())[idx] in (selected_features or [])]
      
    if not ctx.triggered:
        return chart_order
    
    triggered_input = ctx.triggered[0]['prop_id']
    triggered_value = ctx.triggered[0]['value']
    
    if 'switch-dropdown' not in triggered_input:
        return chart_order
    
    try:
        dropdown_id = json_lib.loads(triggered_input.split('.')[0])
        dropdown_index = dropdown_id['index']
    except Exception as e:
        print(f"Error parsing dropdown ID: {e}")
        print(f"Triggered input: {triggered_input}")
        return chart_order
        
    features_list = list(features.keys())
    if dropdown_index is None or not triggered_value or triggered_value == "switch":
        return chart_order
    if triggered_value not in features_list:
        return chart_order

    target_index = features_list.index(triggered_value)

    if dropdown_index == target_index:
        return chart_order

    new_order = chart_order.copy() if isinstance(chart_order, list) else []

    try:
        pos_a = new_order.index(dropdown_index)
    except ValueError:
        return chart_order
    try:
        pos_b = new_order.index(target_index)
    except ValueError:
        return chart_order

    new_order[pos_a], new_order[pos_b] = new_order[pos_b], new_order[pos_a]
    return new_order

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