from dash import Dash, Input, Output, State, html, dcc, ctx
import plotly.express as px
from radviz_plotly import RadViz2D
import pandas as pd
from data import load_debt_data, total_annual_debt, total_annual_unemployment, filter_by_year, filter_by_years, filter_by_states, combine_vars
import json
import math

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
                    projection="mercator",
                    title="Germany Economic Indicators Map"
                   )
germany_map.update_geos(fitbounds="locations", visible=False)
germany_map.update_layout(
    margin={"r":0,"t":40,"l":0,"b":0},  # Added top margin for title
    title={
        "text": "Germany Economic Indicators Map",
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
    
    # Make the title bold
    bar_chart.update_layout(
        title={
            'text': f'{title} in {year}',
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
        markers=True  # Add markers for better visibility
    )
    
    # Make the title bold
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
    
    # Add a vertical line to track the current year if provided
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

# Radviz chart
def get_radviz(title, data, selected_states, current_year=None):
    filtered_data = filter_by_states(data, selected_states)
    filtered_data = filter_by_year(filtered_data, current_year)

    y = filtered_data['state']
    x = filtered_data.drop(['state'], axis=1) 

    BPs = 10000
    radviz = RadViz2D(y, x, BPs)

    return radviz

# LAYOUT
app.layout = html.Div(children=[
    html.H1(children='German Debt and Socioeconomic Factors'),
    html.Div(
        id='content',
        children=[
            html.Div(
                id='map-container',
                children=[
                    dcc.Graph(id='debt-map', figure=germany_map)
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
                    # Line charts with navigation
                    html.Div(
                        id='line-chart-section',
                        className='chart-section',
                        children=[
                            html.Div(className='chart-header', children=[
                                html.H3("Radviz Test", className='chart-title'),
                                # html.Div(className='chart-navigation', children=[
                                #     html.Button("◀", id="prev-line-chart", className="nav-button"),
                                #     html.Span(id="line-chart-page-indicator", children="1/1"),
                                #     html.Button("▶", id="next-line-chart", className="nav-button"),
                                # ])
                            ]),
                            html.Div(id="radviz-container", className="chart-container")
                        ]
                    ),
                    # Bar charts with navigation
                    html.Div(
                        id='bar-chart-section',
                        className='chart-section',
                        children=[
                            html.Div(className='chart-header', children=[
                                html.H3("Bar Charts", className='chart-title'),
                                html.Div(className='chart-navigation', children=[
                                    html.Button("◀", id="prev-bar-chart", className="nav-button"),
                                    html.Span(id="bar-chart-page-indicator", children="1/1"),
                                    html.Button("▶", id="next-bar-chart", className="nav-button"),
                                ])
                            ]),
                            html.Div(id="bar-charts-container", className="chart-container")
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

# Callback to handle map clicks for state selection
@app.callback(
    Output("state-dropdown", "value"),
    Input("debt-map", "clickData"),
    State("state-dropdown", "value")
)
def update_state_selection(clickData, current_selection):
    if clickData is None:
        return current_selection
    
    # Get clicked state
    state_clicked = clickData["points"][0]["location"]
    
    # Initialize selected states
    selected = current_selection.copy() if current_selection else []
    
    # Toggle selection (add if not present, remove if present)
    if state_clicked in selected:
        selected.remove(state_clicked)
    else:
        selected.append(state_clicked)
        
    return selected

# Store current page index for charts
line_chart_page = 0
bar_chart_page = 0

@app.callback(
    [Output("radviz-container", "children"),],
    [Input("state-dropdown", "value"),
     Input("feature-checklist", "value"),
     Input("time-slider", "min"),
     Input("time-slider", "max"),
     Input("time-slider", "value")],
)
def update_radviz(selected_states, selected_features, min_year, max_year, current_year):
    # Get the callback context
    triggered = ctx.triggered_id

    data = combine_vars()
    filtered_data = filter_by_years(data, min_year, max_year)
    filtered_data = filter_by_states(filtered_data, selected_states)
    
    chart = get_radviz('Radviz test', filtered_data, selected_states, current_year)

    
    return chart

@app.callback(
    [Output("bar-charts-container", "children"),
     Output("bar-chart-page-indicator", "children")],
    [Input("time-slider", "value"),
     Input("state-dropdown", "value"),
     Input("feature-checklist", "value"),
     Input("prev-bar-chart", "n_clicks"),
     Input("next-bar-chart", "n_clicks")],
    [State("bar-chart-page-indicator", "children")]
)
def update_bar_charts(year, selected_states, selected_features, prev_clicks, next_clicks, current_page_indicator):
    # Create all charts first
    all_charts = []
    for title, data in features.items():
        if title not in selected_features:
            continue
        chart = get_bar_chart(title, data, year, selected_states)
        all_charts.append(dcc.Graph(figure=chart))
    
    # Calculate total pages and handle pagination
    total_charts = len(all_charts)
    if total_charts == 0:
        return [html.Div("No charts to display")], "0/0"
    
    charts_per_page = 1
    total_pages = max(1, math.ceil(total_charts / charts_per_page))
    
    # Get current page
    global bar_chart_page
    triggered = ctx.triggered_id
    
    if triggered == 'prev-bar-chart':
        bar_chart_page = (bar_chart_page - 1) % total_pages
    elif triggered == 'next-bar-chart':
        bar_chart_page = (bar_chart_page + 1) % total_pages
    elif triggered == 'feature-checklist':
        # Reset to first page when features change
        bar_chart_page = 0
        
    # Ensure page is valid
    bar_chart_page = max(0, min(bar_chart_page, total_pages - 1))
    
    # Get charts for current page
    start_idx = bar_chart_page * charts_per_page
    end_idx = min(start_idx + charts_per_page, total_charts)
    current_charts = all_charts[start_idx:end_idx]
    
    # Update page indicator
    page_indicator = f"{bar_chart_page + 1}/{total_pages}"
    
    return current_charts, page_indicator

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