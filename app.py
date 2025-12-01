from dash import Dash, Input, Output, State, html, dcc, ctx, no_update, ALL
from dash.exceptions import PreventUpdate
import plotly.express as px
import pandas as pd
from data import load_debt_data, total_annual_debt, total_annual_unemployment, filter_by_year, filter_by_years, filter_by_states, population_from_density, normalized_debt_per_capita, normalized_unemployment_per_capita, load_recipients_of_benefits, load_graduation_rates, get_dataset_unit, load_expenditure_on_public_schools, combine_features, normalize_recipients_of_benefits_state_per_1000_inhabitants, normalize_tourism_per_capita    
import json
import math
import os
import numpy as np
import json as json_lib
import plotly.graph_objects as go
import plotly.colors as pc

# Global Variable
TIMEWHEEL_JUST_UPDATED = False

app = Dash()

features = {
    "Debt": normalized_debt_per_capita(),
    "Unemployment": normalized_unemployment_per_capita(),
    "Graduation Rates": load_graduation_rates(),
    "Recipients of Benefits": normalize_recipients_of_benefits_state_per_1000_inhabitants(),
    "Expenditure on Public Schools": load_expenditure_on_public_schools(),
}

def get_features(feature_names):
    selected_features = []
    selected_features.append(features.get("Debt"))
    for name in feature_names:
        if name != "Debt" and name in features:
            selected_features.append(features.get(name))
    return selected_features

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
        "text": "Debt averaged 2010 - 2023 (EUR per capita)",
        "y": 0.93,
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
#TODO : dynamic title
secondary_map.update_layout(
    margin={"r":0,"t":40,"l":0,"b":0}, 
    title={
        "text": "Unemployment by State",
        "y": 0.93,
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
#TODO : dynamic title
difference_map.update_layout(
    margin={"r":0,"t":40,"l":0,"b":0}, 
    title={
        "text": "Difference in Debt and Unemployment",
        "y": 0.93,
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
    marks={year: {'label': str(year), 'style': {'transform': 'rotate(-45deg)', 'font-size': '12px'}} for year in range(min_year, max_year + 1, label_step_size)}
)

range_slider = dcc.RangeSlider(
    id='time-range-slider',
    min=min_year,
    max=max_year,
    step=1,
    value=[min_year, max_year],
    marks={year: {'label': str(year), 'style': {'transform': 'rotate(-45deg)', 'font-size': '12px'}} for year in range(min_year, max_year + 1, label_step_size)}
)

# TIMEWHEEL
def draw_line(fig, x1, y1, x2, y2, mode='trace', color='rgb(0,0,0)', width=1, opacity=1, metadata=[], label="", start_label="", end_label="", hasArrowTip=False, start_end_label_distance=0.05, axis_swapped=False):
    label_distance = 0.3
    
    if mode == "trace":
        if metadata: # data that is displayed when hovering over datapoint
            template = """
            %{customdata[3]}: %{customdata[0]} %{customdata[4]}<br>
            Debt: %{customdata[5]} %{customdata[6]}<br>
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
                       hoverinfo='skip'
                       )
        )
        # slightly cursed line hovering
        if metadata:
            hover_line_width = max(6, width + 4)
            x = np.linspace(x1, x2, num=10)
            y = np.linspace(y1, y2, num=10)
            meta_copied = [metadata for _ in range(10)]
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    line=dict(color='rgba(0,0,0,0)', width=hover_line_width),
                    opacity=0.001,
                    customdata=meta_copied,
                    hovertemplate=template,
                    showlegend=False
                )
            )

    elif mode == "shape":
        fig.add_annotation(
            x=x2, 
            y=y2, 
            ax=x1,
            ay=y1, 
            xref='x',
            yref='y',
            axref='x',
            ayref='y',
            text='',  # show only arrow
            showarrow=hasArrowTip,
            arrowhead=5,
            arrowsize=0.5,
            arrowwidth=width,
            arrowcolor=color,
        )
    elif mode == "dotted":
        fig.add_shape(
            type="line",
            x0=x1,
            y0=y1,
            x1=x2,
            y1=y2,
            line=dict(color=color, width=width, dash="dot"),
            opacity=opacity
        )
    else:
        raise ValueError("Invalid Mode")
    
    # add labels
    x_dir = x2-x1
    y_dir = y2-y1
    
    if axis_swapped:
        x_offset_dir = y_dir
        y_offset_dir = -x_dir
    else:
        x_offset_dir = -y_dir
        y_offset_dir = x_dir
        
    if label:
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
    if start_label:
        label_x = x1 - x_dir * start_end_label_distance
        label_y = y1 - y_dir * start_end_label_distance
        
        fig.add_annotation(
            x=label_x,
            y=label_y,
            text=start_label,
            showarrow=False,
            arrowhead=0,
            font=dict(size=8, color=color)
        )
    if end_label:
        label_x = x2 + x_dir * start_end_label_distance
        label_y = y2 + y_dir * start_end_label_distance
        
        fig.add_annotation(
            x=label_x,
            y=label_y,
            text=end_label,
            showarrow=False,
            arrowhead=0,
            font=dict(size=8, color=color)
        )
        
    
def get_timewheel(data, selected_indices, bundling_mode="none"):
    fig = go.Figure()

    radius = 1
    center_line_width = 0.5
    axis_gap = 0.1
    debt_label_distance = 0.5
    metadata_cols = ["state", "year"]  
    
    # do bundling
    if bundling_mode == "state":
        data = data.groupby("state", as_index=False).agg(
            {col: lambda x: round(x.mean(), 2) for col in data.columns if col not in metadata_cols}
        )
        data["year"] = "Bundled"
        
    elif bundling_mode == "year":
        data = data.groupby("year", as_index=False).agg(
            {col: lambda x: round(x.mean(), 2) for col in data.columns if col not in metadata_cols}
        )
        data["state"] = "Bundled"
    
    # split up dataframe
    debt_data = data["Debt"]
    metadata = data[metadata_cols]
    features_data = data.drop(columns=["Debt"] + metadata_cols)
    
    num_features = len(features_data.columns)
    angle_interval = 2*math.pi / num_features
    
    # colors = [pc.sample_colorscale("viridis", i/num_features+1e-10)[0] for i in range(num_features)]
    colors = [px.colors.qualitative.Safe[i] for i in range(num_features)]
    # hard coded colours picked from  colorgorical shown in lecture 2 as a suggestion
    # colors = ['rgb(53,97,143)', 'rgb(111,239,112)', 'rgb(118,30,126)', 'rgb(133,194,212)']
    
    # draw center line
    draw_line(
            fig,
            -center_line_width,
            0,
            center_line_width,
            0, 
            mode="shape",
            width=3,
            hasArrowTip=True,
            start_label=str(int(debt_data.min())),
            end_label=str(int(debt_data.max())),
            start_end_label_distance=0.15
        )
    
    debt_normalized = (debt_data - debt_data.min()) / (debt_data.max()-debt_data.min())
        
    current_angle = math.pi/2
    current_point = 0
    
    min_x = 1
    
    # draw feature axes
    for i in range(num_features): 
        
        feature_data = features_data.iloc[:, i]
        
        #for n = 1,2, handle positions manually
        if num_features <= 2:
            if i == 0:
                start_x = -0.5
                start_y = 0.5
                end_x = 0.5
                end_y = 0.5   
            elif i == 1:
                start_x = 0.5
                start_y = -0.5
                end_x = -0.5
                end_y = -0.5
        # for larger n, calculate positions on 
        else:
            start_x = math.cos(current_angle + angle_interval) * radius
            start_y = math.sin(current_angle + angle_interval) * radius
            
            end_x = math.cos(current_angle) * radius
            end_y = math.sin(current_angle) * radius
        
        axis_dir_x = end_x - start_x
        axis_dir_y = end_y - start_y
        
        start_x += axis_dir_x * axis_gap
        start_y += axis_dir_y * axis_gap
        end_x -= axis_dir_x * axis_gap
        end_y -= axis_dir_y * axis_gap

        # Ensure axes run from left to right and flag
        axis_swapped = False
        if end_x < start_x:
            start_x, end_x = end_x, start_x
            start_y, end_y = end_y, start_y
            axis_swapped = True
        
        min_x = min(min_x, start_x, end_x)
    
        draw_line(
            fig,
            start_x,
            start_y,
            end_x,
            end_y,
            color=colors[i],
            mode="shape",
            width=3,
            label=features_data.columns[i],
            start_label=str(feature_data.min()),
            end_label=str(feature_data.max()),
            hasArrowTip=True,
            axis_swapped=axis_swapped
        )
        
        # draw datapoints
        feature_normalized = (feature_data - feature_data.min()) / (feature_data.max()-feature_data.min())

        for j, feature_datapoint in enumerate(feature_normalized):
            datapoint_start_x = start_x + (end_x - start_x) * feature_datapoint
            datapoint_start_y = start_y + (end_y - start_y) * feature_datapoint
            
            datapoint_end_x = -center_line_width + 2*center_line_width*debt_normalized.iloc[j]
            datapoint_end_y = 0

            opacity = 0.8
            labels = []
            for k, col in enumerate(metadata.columns):
                if bundling_mode == col:
                    labels.append("Bundled")
                else:
                    labels.append(metadata.iloc[j,k])
                
            # if current_point not in selected_indices:
            #     opacity *= 0.2 

            if (metadata.iloc[j,0], metadata.iloc[j,1]) not in selected_indices:
                opacity *= 0.2

                
            if bundling_mode == "none":
                opacity *= 0.5
                
            feature_name = features_data.columns[i]
            feature_unit = get_dataset_unit(feature_name, features)
            debt_value = debt_data.iloc[j]
            debt_unit = get_dataset_unit("Debt", features)
            
            draw_line(
                fig,
                datapoint_start_x,
                datapoint_start_y,
                datapoint_end_x,
                datapoint_end_y,
                color=colors[i],
                opacity=opacity,
                metadata=[feature_data.iloc[j],  metadata.iloc[j,0], metadata.iloc[j,1], feature_name, feature_unit, debt_value, debt_unit],
                width=4 if not bundling_mode == "none"else 1
            )

            current_point += 11
        
        current_angle += angle_interval
        
        # draw center label
        draw_line(
            fig,
            -center_line_width,
            0,
            min_x - debt_label_distance,
            0,
            mode="dotted",
            opacity=0.2,
            width=0.5,
            end_label="Debt"
        )
    
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
                            ]),
                            dcc.Dropdown(
                                options=[
                                    {'label': 'No Bundling', 'value': 'none'},
                                    {'label': 'Bundling by State', 'value': 'state'},
                                    {'label': 'Bundling by Year', 'value': 'year'}
                                ],
                                value='none',
                                clearable=False,
                                id='bundling-mode-switch-dropdown',
                                style={'flex': '0 0 auto', 'height': '30px', 'font-size': '10px', 'padding': '5px 8px', 'margin-top': '10px'}
                            )
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
                                        df['state'].unique(),
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
                                                        [feature for feature in list(features.keys()) if feature != "Debt"],
                                                        [feature for feature in list(features.keys()) if feature != "Debt"],
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
    dcc.Store(id='timewheel-selection-store'),
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


def get_single_title(map_feature, year):
    unit = get_dataset_unit(map_feature, features)
    if map_feature == "Debt":
        title = f"{map_feature} in {year} ({unit})"
    elif map_feature == "Expenditure on Public Schools":
        title = f"{map_feature} <br>in {year} ({unit})"
    else:
        title = f"{map_feature} in {year}<br>({unit})"
    return title
def get_average_title(map_feature, start, end):
    unit = get_dataset_unit(map_feature, features)
    if map_feature == "Debt":
        title = f"{map_feature} averaged {start} - {end} ({unit})"
    elif map_feature == "Expenditure on Public Schools":
        title = f"{map_feature} averaged <br>{start} - {end} ({unit})"
    else:
        title = f"{map_feature} averaged {start} - {end}<br>({unit})"
    return title

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

    if slider_mode == "single":
        year = single_value if single_value is not None else single_min
        filtered = filter_by_year(data_df, year)
        title = get_single_title(map_feature, year)
    else:
        start, end = (range_value if range_value and len(range_value) == 2 else (single_min, single_max))
        filtered = filter_by_years(data_df, start, end)
        filtered = filtered.groupby("state", as_index=False)["value"].mean()
        title = get_average_title(map_feature, start, end)
        

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

    state_data["bin"] = pd.qcut(filtered["value"], q=5, duplicates='drop', precision=1)
    c_order = list(state_data["bin"].cat.categories)
    fig = px.choropleth(
        state_data,
        geojson=germany_geojson,
        locations="state",
        featureidkey="properties.NAME_1",
        color="bin",
        category_orders={"bin": c_order},
        hover_data={"value": True},
        color_discrete_sequence=px.colors.sequential.Inferno_r,
        projection="mercator",
        title="Germany Economic Indicators Map"
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0},
        title={
            "text": title,
            "y": 0.93,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 16}
        },
        legend=dict(
            title=None,
            x=1.20,
            y=0,
            xanchor='right',
            yanchor='bottom',
            bgcolor='rgba(255,255,255,0)',
        )
    )
    return fig

@app.callback(
    Output("secondary-map", "figure"),
    Input("time-slider", "value"),
    Input("time-slider", "min"),
    Input("time-slider", "max"),
    Input("time-range-slider", "value"),
    Input("time-slider-mode-store", "data"),
    Input("secondary-feature-dropdown", "value")
)
def update_secondary_map(single_value, single_min, single_max, range_value, slider_mode, map_feature):
    secondary_data_df = features.get(map_feature)
    if slider_mode == "single":
        year = single_value if single_value is not None else single_min
        filtered = filter_by_year(secondary_data_df, year)
        title = get_single_title(map_feature, year)
    else:
        start, end = (range_value if range_value and len(range_value) == 2 else (single_min, single_max))
        filtered = filter_by_years(secondary_data_df, start, end)
        filtered = filtered.groupby("state", as_index=False)["value"].mean()
        title = get_average_title(map_feature, start, end)
    
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
    if map_feature == "Recipients of Benefits":
        state_data["bin"] = pd.qcut(filtered["value"], q=5, duplicates='drop', precision=1)
        c_order = list(state_data["bin"].cat.categories)
        fig = px.choropleth(
            state_data,
            geojson=germany_geojson,
            locations="state",
            featureidkey="properties.NAME_1",
            color="bin",
            category_orders={"bin": c_order},
            hover_data={"value": True},
            color_discrete_sequence=px.colors.sequential.Inferno_r,
            projection="mercator",
            title="Germany Economic Indicators Map"
        )
    else: 
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
            "y": 0.93,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 16}
        },
        legend=dict(
            title=None,
            x=1.05,
            y=0,
            xanchor='right',
            yanchor='bottom',
            bgcolor='rgba(255,255,255,0)',
        )
    )
    return fig

@app.callback(
    Output("difference-map", "figure"),
    Input("time-slider", "value"),
    Input("time-slider", "min"),
    Input("time-slider", "max"),
    Input("time-range-slider", "value"),
    Input("time-slider-mode-store", "data"),
    Input("secondary-feature-dropdown", "value")
)
def update_difference_map(single_value, single_min, single_max, range_value, slider_mode, map_feature):
    debt_data_df = features.get("Debt")
    other_data_df = features.get(map_feature)

    if slider_mode == "single":
        year = single_value if single_value is not None else single_min
        filtered_debt = filter_by_year(debt_data_df, year)
        filtered_other = filter_by_year(other_data_df, year)
        title = f"Difference in Debt and {map_feature} in {year}"
    else:
        start, end = (range_value if range_value and len(range_value) == 2 else (single_min, single_max))
        filtered_debt = filter_by_years(debt_data_df, start, end)
        filtered_debt = filtered_debt.groupby("state", as_index=False)["value"].mean()
        filtered_other = filter_by_years(other_data_df, start, end)
        filtered_other = filtered_other.groupby("state", as_index=False)["value"].mean()
        title = f"Difference in Debt and {map_feature} averaged {start} - {end}"
    if filtered_debt.empty or filtered_other.empty:
        df_difference = pd.DataFrame({
            "state": ["Berlin"],
            "value": [0]
        })
        title = "No data available for Difference Map"
    else:
        #Align tables by state
        filtered_debt = filtered_debt.set_index("state")
        filtered_other = filtered_other.set_index("state")
        debt_scaled = (filtered_debt["value"] - filtered_debt["value"].min()) / (filtered_debt["value"].max() - filtered_debt["value"].min())
        other_scaled = (filtered_other["value"] - filtered_other["value"].min()) / (filtered_other["value"].max() - filtered_other["value"].min())

        df_difference = filtered_debt.copy()
        df_difference["value"] = debt_scaled - other_scaled
        #Insert index for figure again
        df_difference = df_difference.reset_index()
    fig = px.choropleth(
        df_difference,
        geojson=germany_geojson,
        locations="state",
        featureidkey="properties.NAME_1",
        color="value",
        projection="mercator",
        title="Difference Map"
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0},
        title={
            "text": title,
            "y": 0.93,
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
        default_marks = {year: {'label': str(year), 'style': {'transform': 'rotate(-45deg)', 'font-size': '12px'}} for year in range(2000, 2021)}
        return 2000, 2020, 2000, default_marks, 2000, 2020, [2000, 2020], default_marks
    
    selected_dfs = get_features(selected_features)
    if not selected_dfs:
        default_marks = {year: {'label': str(year), 'style': {'transform': 'rotate(-45deg)', 'font-size': '12px'}} for year in range(2000, 2021)}
        return 2000, 2020, 2000, default_marks, 2000, 2020, [2000, 2020], default_marks
    
    if len(selected_features) == 1:
        df = selected_dfs[0]
        min_year = int(df['year'].min())
        max_year = int(df['year'].max())
        marks = {year: {'label': str(year), 'style': {'transform': 'rotate(-45deg)', 'font-size': '12px'}} for year in range(min_year, max_year + 1)}
    else:
        min_years = [int(df['year'].min()) for df in selected_dfs]
        max_years = [int(df['year'].max()) for df in selected_dfs]
        min_year = max(min_years)
        max_year = min(max_years)
        
        if min_year > max_year:
            default_marks = {year: {'label': str(year), 'style': {'transform': 'rotate(-45deg)', 'font-size': '12px'}} for year in range(2000, 2021)}
            return 2000, 2020, 2000, default_marks, 2000, 2020, [2000, 2020], default_marks
        marks = {year: {'label': str(year), 'style': {'transform': 'rotate(-45deg)', 'font-size': '12px'}} for year in range(min_year, max_year + 1)}
    
    single_value = current_single_year if min_year <= current_single_year <= max_year else min_year
    
    if current_range_value and len(current_range_value) == 2:
        range_start = max(min_year, min(current_range_value[0], max_year))
        range_end = min(max_year, max(current_range_value[1], min_year))
        range_value = [range_start, range_end]
    else:
        range_value = [min_year, max_year]
    
    return (min_year, max_year, single_value, marks, 
            min_year, max_year, range_value, marks)
    
previous_selection = []
@app.callback(
    Output("feature-checklist", "value"),
    Input("feature-checklist", "value")
)
def update_feature_checklist(selected_features):
    
    # limit to at least one feature selected
    global previous_selection
    
    if len(selected_features) == 1:
        previous_selection = selected_features
        
    elif len(selected_features) == 0:
        selected_features = previous_selection if previous_selection else ["Unemployment"]
        
    return selected_features

@app.callback(
    Output("timewheel", "figure"),
    Input("feature-checklist", "value"),
    Input("state-dropdown", "value"),
    Input("time-slider", "value"),
    Input("time-range-slider", "value"),
    Input("time-slider-mode-store", "data"),
    Input("time-slider", "min"),
    Input("time-slider", "min"),
    Input("timewheel-selection-store", "data"),
    Input("bundling-mode-switch-dropdown", "value")
)
def update_time_wheel(selected_features, selected_states, single_value, range_value, slider_mode, single_min, single_max, selected_data, bundling_mode):

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
        selected_indices = [ (p['customdata'][1], p['customdata'][2]) for p in selected_data["points"] ]
        if len(selected_indices) != 0:
            TIMEWHEEL_JUST_UPDATED = True
    else:
        selected_indices = []

    timewheel = get_timewheel(filtered_data, selected_indices, bundling_mode)
    return timewheel


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)