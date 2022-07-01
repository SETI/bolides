import io
import flask
from dash import Dash, html, dcc
from dash import dash_table
import plotly.express as px
import pandas as pd
from bolides import BolideDataFrame

import datetime
import numpy as np

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

server = flask.Flask(__name__)
app = Dash(__name__, server=server)
app.title = 'bolides webapp'

def get_empty_map():
    fig = px.scatter_mapbox(lat=[], lon=[], mapbox_style="open-street-map", zoom=0)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=900)
    return fig

def get_map(df, boundary, color_column):
    import numbers
    numeric_columns=['datetime']
    too_long = ['otherInformation', 'reason']
    if len(df)>0:
        for col in df.columns:
            if isinstance(df[col][0], numbers.Number) and (col not in too_long):
                numeric_columns.append(col)

    fig = px.scatter_mapbox(df, lat="latitude", lon="longitude",
                            mapbox_style="open-street-map",
                            zoom=0, hover_data=numeric_columns,
                            color=color_column)

    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=900)
    from bolides.fov_utils import get_boundary
    import pyproj
    from geopandas import GeoDataFrame
    polygons = get_boundary(boundary)
    if type(polygons) is not list:
        polygons=[polygons]
    aeqd = pyproj.Proj(proj='aeqd', ellps='WGS84', datum='WGS84', lat_0=90, lon_0=0).srs
    gdf = GeoDataFrame(geometry=polygons, crs=aeqd)
    gdf = gdf.to_crs('epsg:4326')
    polygons = gdf.geometry
    import plotly.graph_objects as go
    for num, polygon in enumerate(polygons):
        lons, lats = polygon.exterior.coords.xy
        lons = np.array(lons)
        lats = np.array(lats)
        if boundary[num] == 'goes' or boundary[num] == 'goes-w':
            lons = lons - (lons>50) * 360

        #lons = [p.x for point in polygon]
        #lats = [p.y for point in polygon]
        fig.add_trace(go.Scattermapbox(
            mode="lines",
            lon = lons,
            lat=lats,
            name=boundary[num]))
    return fig

def generate_table(dataframe, max_rows=10):
    print('hmm')
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

colors = {
    'background': '#000000',
    'text': '#FFFFFF'
}

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
def get_df(source, filter_query=None):
    import os
    filename = source+'_'+datetime.datetime.today().strftime('%Y%m%d')+'.csv'
    if source not in ['usg', 'website']:
        pass
    elif os.path.isfile(filename):
        print(filename,'exists')
        df = BolideDataFrame(source='csv', files=filename, annotate=False)
    else:
        df = BolideDataFrame(source)
        for col in df.columns:
            if type(df[col][0]) in [list, dict]:
                print('changing '+col+' to str')
                df[col] = [str(x) for x in df[col]]
        df.to_csv(filename)
    unsafe_cols = ['geometry', 'attachments', 'images','csv', 'brightness', 'groundTrack']
    for col in unsafe_cols:
        if col in df.columns:
            df = df.drop(col, axis=1)
    for col in df.columns:
        if type(df[col][0]) is np.float64:
            df[col] = np.around(df[col],3)

    if filter_query is not None:
        filtering_expressions = filter_query.split(' && ')
        for filter_part in filtering_expressions:
            col_name, operator, filter_value = split_filter_part(filter_part)

            if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
                df = df.loc[getattr(dff[col_name], operator)(filter_value)]
            elif operator == 'contains':
                df = df.loc[dff[col_name].str.contains(filter_value)]
            elif operator == 'datestartswith':
                df = df.loc[dff[col_name].str.startswith(filter_value)]

    return df

df = get_df('website')

app.layout = html.Div(style={}, children=[
    html.H1(
        children='bolides 🌠 webapp',
        style={'textAlign': 'center',
               }),
              

    html.Div(children='''
        A webapp to explore bolide data.
        ''', 
        style={'textAlign': 'left',
               }),

    dcc.Dropdown(['usg', 'website'], 'usg', id='source-select'),

    dcc.Download(id="download"),
    html.Button("Export as csv",
                id="save-button"),
    #html.Button("Filter data from map selection",
    #            id="map-filter"),

    dcc.Checklist(['goes','goes-e','goes-w', 'fy4a-n','fy4a-s'], [], inline=True, id='boundary-checklist'),

    dcc.Dropdown(['latitude','longitude','solarhour','sun_alt'], 'latitude', id='color'),
    dcc.Graph(
        id='main-map',
        figure=get_empty_map()
    ),

    dash_table.DataTable(df.to_dict('records'),
        id='main-table',
        columns=[
            {"name": i, "id": i, "deletable": True, "selectable": True} for i in df.columns
        ],
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        column_selectable="single",
        row_selectable="multi",
        row_deletable=True,
        selected_columns=[],
        selected_rows=[],
        page_action="native",
        page_current=0,
        page_size=50,
        style_table={'overflowX': 'scroll'},
    ),
    dcc.Dropdown(['latitude','longitude','solarhour','sun_alt'], 'longitude', id='scatter-x'),
    dcc.Dropdown(['latitude','longitude','solarhour','sun_alt'], 'latitude', id='scatter-y'),
    dcc.Graph(
        id='scatter',
        figure=px.scatter(x=df.longitude, y=df.latitude)
    ),
    dcc.Dropdown(['latitude'], 'latitude', id='hist-var'),
    dcc.Input(type='number', min=1, max=500, value=30, id='hist-bins'),
    dcc.Graph(
        id='hist',
        figure=px.histogram(df.latitude)
    ),
    dcc.Graph(
        id='lightcurve',
        figure={})

])

@app.callback(
Output('main-table', 'data'),
Output('main-table', 'columns'),
Output('scatter-x', 'options'),
Output('scatter-y', 'options'),
Output('hist-var', 'options'),
Output('color', 'options'),
Input('source-select', 'value')
)
def change_source(source):
    print('changing source')
    df = get_df(source)
    #update_map(df)
    data = df.to_dict('records')
    columns=[{"name": i, "id": i, "deletable": True, "selectable": True} for i in df.columns]
    numeric_columns = ['datetime']
    color_columns = []
    import numbers
    if len(df)>0:
        for col in df.columns:
            if isinstance(df[col][0], numbers.Number):
                numeric_columns.append(col)
                color_columns.append(col)
            elif len(df[col].unique()) < 20:
                color_columns.append(col)

    return data, columns, numeric_columns, numeric_columns, numeric_columns, color_columns

@app.callback(
Output('scatter', 'figure'),
Input('scatter-x', 'value'),
Input('scatter-y', 'value'),
Input('color', 'value'),
Input('main-table', 'filter_query'),
Input('source-select', 'value')
)
def update_scatter(x, y, color_column, filter_query, source):
    df = get_df(source, filter_query)
    fig = px.scatter(df, x=x, y=y, color=color_column)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig

@app.callback(
Output('hist', 'figure'),
Input('hist-var', 'value'),
Input('hist-bins', 'value'),
Input('main-table', 'filter_query'),
Input('source-select', 'value')
)
def update_hist(var, bins, filter_query, source):
    print('updating histogram')
    df = get_df(source, filter_query)
    fig = px.histogram(df, x=var, nbins=bins)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig

@app.callback(
Output("download", "data"),
Input("save-button", "n_clicks"),
State('main-table', 'filter_query'),
State('source-select', 'value'))
def download_as_csv(n_clicks, filter_query, source):
    df = get_df(source, filter_query)
    if not n_clicks:
      raise PreventUpdate
    download_buffer = io.StringIO()
    df.to_csv(download_buffer, index=False)
    download_buffer.seek(0)
    if len(df) == 0:
        filename = "column_names.csv"
    else:
        dt = datetime.datetime.fromisoformat(df.date_retrieved[0])
        datestr = dt.strftime("%Y%m%d%H%M%S")
        filename = df.source[0]+'_'+datestr+'.csv'
    return dict(content=download_buffer.getvalue(), filename=filename)

operators = [['ge ', '>='],
             ['le ', '<='],
             ['lt ', '<'],
             ['gt ', '>'],
             ['ne ', '!='],
             ['eq ', '='],
             ['contains '],
             ['datestartswith ']]


def split_filter_part(filter_part):
    for operator_type in operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find('{') + 1: name_part.rfind('}')]

                value_part = value_part.strip()
                v0 = value_part[0]
                if (v0 == value_part[-1] and v0 in ("'", '"', '`')):
                    value = value_part[1: -1].replace('\\' + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part

                # word operators need spaces after them in the filter string,
                # but we don't want these later
                return name, operator_type[0].strip(), value

    return [None] * 3


@app.callback(
Output("main-map", "figure"),
Input("main-table", "filter_query"),
Input("source-select", "value"),
Input("boundary-checklist", "value"),
Input("color", "value"))
def update_map(filter_query, source, boundary, color_column):
    df = get_df(source, filter_query)

    fig = get_map(df, boundary, color_column)
    return fig

def update_table_style(selectedData):
    table_style_conditions = []
    if selectedData != None:
        points_selected = []
        for point in selectedData['points']:
            points_selected.append(point['pointIndex'])
        selected_styles = [{'if': {'row_index': i},
                            'backgroundColor': 'pink'} for i in points_selected]
        table_style_conditions.extend(selected_styles)
    return table_style_conditions

@app.callback(
Output("main-table", "style_data_conditional"),
Input("main-map", "selectedData")
)
def color_from_map(selectedData):
    style = update_table_style(selectedData)
    return style

@app.callback(
Output('lightcurve', 'figure'),
Input('main-map', 'clickData'),
Input("main-table", "filter_query"),
Input("source-select", "value")
)
def update_lightcurve(clickData, filter_query, source):
    import plotly.graph_objects as go
    print('updating lightcurve')
    df = get_df(source, filter_query).iloc[clickData['points'][0]['pointIndex']]
    df = pd.DataFrame(df).T
    print(type(df))
    print(df)
    df.to_csv('tmp.csv')
    bdf = BolideDataFrame('csv', 'tmp.csv')
    print(bdf)
    bdf.add_website_data()
    print('getting light curve')
    print(len(bdf.lightcurves))
    lcc = bdf.lightcurves.iloc[0]
    dfs = []
    for lc in lcc:
        dfs.append(pd.DataFrame({'time':lc.time.value, 'flux':lc.flux.value, 'source':lc.meta['LABEL']}))
        #figs.append(px.line(x=lc.time.value, y=lc.flux.value))
    lcc_df = pd.concat(dfs)
    fig = px.line(lcc_df, x='time', y='flux', color='source')
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, hovermode='x unified')
    return fig

    


# @app.callback(
# Output("main-table", "data"),
# Input("map-filter", "n_clicks"),
# State("main-table", "derived_virtual_data"),
# State("main-map", "selectedData")
# )
# def filter_from_map(n_clicks, data, selectedData):
#     df = pd.DataFrame.from_dict(data)
#     idx = [p['pointIndex'] for p in selectedData['points']]
#     return df.iloc[idx].to_dict('records')

if __name__ == '__main__':
    app.run_server(debug=False)
