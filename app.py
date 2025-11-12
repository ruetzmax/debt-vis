from dash import Dash, Input, Output, State, html, dcc, ctx, ALL
from dash.exceptions import PreventUpdate
import plotly.express as px
import pandas as pd
from data import load_debt_data, total_annual_debt, total_annual_unemployment, filter_by_year, filter_by_years, filter_by_states, population_from_density, normalized_debt_per_capita, normalized_unemployment_per_capita, load_recipients_of_benefits, load_graduation_rates, get_dataset_unit, load_expenditure_on_public_schools, combine_features
import json
import math
import json as json_lib
import plotly.graph_objects as go
import plotly.colors as pc

# Global Variable
TIMEWHEEL_JUST_UPDATED = False

app = Dash()

features = {
    "Debt": normalized_debt_per_capita(),
    "Duimmy": normalized_debt_per_capita(),
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

# TIME SLIDER - calculate intersection of all features at startup
min_years = []
max_years = []
label_step_size = 10

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
            marks={year: str(year) for year in range(min_year, max_year + 1, label_step_size)}
        )

range_slider = dcc.RangeSlider(
            id='time-range-slider',
            min=min_year,
            max=max_year,
            step=1,
            value=[min_year, max_year],
            marks={year: str(year) for year in range(min_year, max_year + 1, label_step_size)}
        )

# TIMEWHEEL
def draw_line(fig, x1, y1, x2, y2, mode='trace', color='rgb(0,0,0)', width=1, opacity=1, metadata=[], label=""):
    label_distance = 0.15
    
    if mode == "trace":
        if metadata: # data that is displayed when hovering over datapoint
            # TODO: add units
            template = """
            Value: %{customdata[0]}<br>
            State: %{customdata[1]}<br>
            Year: %{customdata[2]}<br>
            <extra></extra>
            """
        else:
            template = ""
            
        fig.add_trace(
            go.Scatter(x=[x1, x2],
                       y=[y1, y2],
                       mode="lines+markers",
                       marker=dict(size=10, opacity=0),
                       line=dict(color=color, width=width),
                       opacity=opacity,
                       customdata=[metadata, metadata],
                       hovertemplate=template
                       )
        )
    elif mode == "shape":
        fig.add_shape(
            type="line",
            x0=x1,
            y0=y1,
            x1=x2,
            y1=y2,
            line=dict(color=color, width=width),
            opacity=opacity
        )
        
        if label:
            x_dir = x2-x1
            y_dir = y2-y1
            
            x_offset_dir = y_dir
            y_offset_dir = -x_dir
            
            label_x = x1 + x_dir*0.5 + x_offset_dir * label_distance
            label_y = y1 + y_dir*0.5 + y_offset_dir * label_distance
            
            # TODO: rotate text (if you dare)?
            # angle = math.degrees(math.atan2(y_dir, x_dir)) 
            angle = 0
            
            fig.add_annotation(
                x=label_x,
                y=label_y,
                text=label,
                textangle=angle,
                showarrow=False,
                arrowhead=0
            )
    else:
        raise ValueError("Invalid Mode")
        
    
def get_timewheel(data, selected_indices):
    fig = go.Figure()

    radius = 1
    center_line_width = 0.3
    
    debt_data = data["Debt"]
    metadata = data[["state", "year"]]
    features_data = data.drop(columns=["Debt", "state", "year"])
    
    num_features = len(features_data.columns)
    angle_interval = 2*math.pi / num_features
    
    colors = [pc.sample_colorscale("Viridis", i/num_features+1e-10)[0] for i in range(num_features)]
    
    # draw center line
    draw_line(
            fig,
            -center_line_width,
            0,
            center_line_width,
            0, 
            mode="shape",
            width=3
        )
    debt_normalized = (debt_data - debt_data.min()) / (debt_data.max()-debt_data.min())
        
    current_angle = math.pi/2
    current_point = 0
    for i in range(num_features):
        # draw feature axis
        start_x = math.cos(current_angle) * radius
        start_y = math.sin(current_angle) * radius
        end_x = math.cos(current_angle + angle_interval) * radius
        end_y = math.sin(current_angle + angle_interval) * radius
    
        draw_line(
            fig,
            start_x,
            start_y,
            end_x,
            end_y, 
            mode="shape",
            width=3,
            label=features_data.columns[i]
        )
        
        # draw datapoints
        feature_data = features_data.iloc[:, i]
        feature_normalized = (feature_data - feature_data.min()) / (feature_data.max()-feature_data.min())

        for j, feature_datapoint in enumerate(feature_normalized):
            datapoint_start_x = start_x + (end_x - start_x) * feature_datapoint
            datapoint_start_y = start_y + (end_y - start_y) * feature_datapoint
            
            datapoint_end_x = -center_line_width + 2*center_line_width*debt_normalized.iloc[j]
            datapoint_end_y = 0

            if current_point in selected_indices:
                opacity = 0.8
            else:
                opacity = 0.2

            draw_line(
                fig,
                datapoint_start_x,
                datapoint_start_y,
                datapoint_end_x,
                datapoint_end_y,
                color=colors[i],
                opacity=opacity,
                metadata=[feature_data.iloc[j], metadata.iloc[j,0], metadata.iloc[j,1]]
            )

            current_point += 1
        
        current_angle += angle_interval
    
    fig.update_layout(
        title="Overview - Debt and Related Factors",
        dragmode="select",
        showlegend=False,
        hovermode="closest",
        plot_bgcolor="rgb(255,255,255)"
    )

    fig.update_xaxes(
        showgrid=False,
        showticklabels=False,
        visible=False
    )

    fig.update_yaxes(
        scaleanchor = "x",
        scaleratio = 0.8,
        showgrid=False,
        showticklabels=False,
        visible=False
    )
    #TODO: make lines hoverable
    
    return fig

timewheel_data = combine_features(features, ["Debt", "Unemployment"])
timewheel = get_timewheel(timewheel_data, [])


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
                        id='timewheel-container',
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
                            html.Div(style={'text-align': 'center'}, children=[
                                dcc.Graph(id='timewheel', figure=timewheel)
                            ])
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
                                    html.Label("Features", style={'font-weight': 'bold', 'margin-bottom': '3px', 'display': 'block', 'font-size': '12px'}),
                                    dcc.Checklist(
                                        list(features.keys()),
                                        [list(features.keys())[0], list(features.keys())[1]],
                                        id="feature-checklist",
                                        style={'font-size': '11px'}
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
                    dcc.Graph(id='debt-map', figure=germany_map, style={'height': '100%'})
                ]
            )
        ]
    ),
    dcc.Store(id='time-slider-mode-store', data='single'),
    dcc.Store(id='timewheel-selection-store'),
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
    if selected_features and len(selected_features) > 0:
        map_feature = selected_features[0]
        data_df = features.get(map_feature)
    else:
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
    Output("timewheel-selection-store", "data"),
    Input("timewheel", "selectedData"),
    prevent_initial_call=True
)
def store_timewheel_selection(selected):
    return selected

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

@app.callback(
    Output("timewheel", "figure"),
    Input("feature-checklist", "value"),
    Input("state-dropdown", "value"),
    Input("time-slider", "value"),
    Input("time-range-slider", "value"),
    Input("time-slider-mode-store", "data"),
    Input("time-slider", "min"),
    Input("time-slider", "min"),
    Input("timewheel-selection-store", "data")
)
def update_time_wheel(selected_features, selected_states, single_value, range_value, slider_mode, single_min, single_max, selected_data):

    global TIMEWHEEL_JUST_UPDATED

    # Prevent updated when selection gets removed
    if TIMEWHEEL_JUST_UPDATED:
        TIMEWHEEL_JUST_UPDATED = False
        raise PreventUpdate


    data = combine_features(features, selected_features)

    if slider_mode == "single":
        year = single_value if single_value else single_min
        filtered_data = filter_by_year(data, year)
    else:
        start, end = (range_value if range_value and len(range_value) == 2 else (single_min, single_max))
        filtered_data = filter_by_years(data, start, end)
        
    filtered_data = filter_by_states(filtered_data, selected_states)

    if selected_data and "points" in selected_data:
        selected_indices = [ p["curveNumber"] for p in selected_data["points"] ]
        if len(selected_indices) != 0:
            TIMEWHEEL_JUST_UPDATED = True
    else:
        selected_indices = []

    timewheel = get_timewheel(filtered_data, selected_indices)
    return timewheel



if __name__ == '__main__':
    app.run(debug=True)