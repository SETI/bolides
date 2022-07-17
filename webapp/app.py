import io
import flask
from dash import Dash, html, dcc
from dash import dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from bolides import BolideDataFrame
from bolides import ShowerDataFrame

import datetime
import numpy as np

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

server = flask.Flask(__name__)
app = Dash(__name__, server=server)
app.title = 'bolide visualizer'

source_dict = {'USG data at https://cneos.jpl.nasa.gov/fireballs/': 'usg',
               'GLM data at https://neo-bolide.ndc.nasa.gov/': 'website',
               'Meteor shower data at https://www.ta3.sk/IAUC22DB/MDC2007/': 'showers'}
projections = ['eckert4', 'GOES-E', 'GOES-W', 'FY4A',
               'airy', 'aitoff', 'albers', 'albers usa', 'august',
               'azimuthal equal area', 'azimuthal equidistant', 'baker',
               'bertin1953', 'boggs', 'bonne', 'bottomley', 'bromley',
               'collignon', 'conic conformal', 'conic equal area', 'conic equidistant',
               'craig', 'craster', 'cylindrical equal area',
               'cylindrical stereographic', 'eckert1', 'eckert2',
               'eckert3', 'eckert4 ', 'eckert5', 'eckert6', 'eisenlohr',
               'equirectangular', 'fahey', 'foucaut', 'foucaut sinusoidal',
               'ginzburg4', 'ginzburg5', 'ginzburg6',
               'ginzburg8', 'ginzburg9', 'gnomonic', 'gringorten',
               'gringorten quincuncial', 'guyou', 'hammer', 'hill',
               'homolosine', 'hufnagel', 'hyperelliptical',
               'kavrayskiy7', 'lagrange', 'larrivee', 'laskowski',
               'loximuthal', 'mercator', 'miller', 'mollweide', 'mt flat polar parabolic',
               'mt flat polar quartic', 'mt flat polar sinusoidal',
               'natural earth', 'natural earth1', 'natural earth2',
               'nell hammer', 'nicolosi', 'orthographic',
               'patterson', 'peirce quincuncial', 'polyconic',
               'rectangular polyconic', 'robinson', 'satellite', 'sinu mollweide',
               'sinusoidal', 'stereographic', 'times',
               'transverse mercator', 'van der grinten', 'van der grinten2',
               'van der grinten3', 'van der grinten4',
               'wagner4', 'wagner6', 'wiechel', 'winkel tripel',
               'winkel3']


def generate_table(dataframe, max_rows=10):
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


def df_from_source(source, filename):
    if source == 'showers':
        df = ShowerDataFrame()
    else:
        df = BolideDataFrame(source)

    for col in df.columns:
        if type(df[col][0]) in [list, dict]:
            df[col] = [str(x) for x in df[col]]
    df.to_csv(filename)
    return df


def get_df(source):
    import os
    source = source_dict[source]
    filename = source+'_'+datetime.datetime.today().strftime('%Y%m%d')+'.csv'

    if os.path.isfile(filename):
        print(filename, 'exists')
        try:
            if source == 'showers':
                df = ShowerDataFrame(source='csv', file=filename)
            else:
                df = BolideDataFrame(source='csv', files=filename, annotate=False)
        except pd.errors.ParserError:
            print('Reading csv failed. Fetching again.')
            df = df_from_source(source, filename)

    else:
        df = df_from_source(source, filename)

    unsafe_cols = ['attachments', 'images', 'csv', 'brightness', 'groundTrack']
    for col in unsafe_cols:
        if col in df.columns:
            df = df.drop(col, axis=1)
    return df


def get_df_from_filters(source, filter_query=None, start_date=None, end_date=None,
                        filter_fov=[], boundary_checklist=[], sort_by=None):
    if source is None:
        return pd.DataFrame({})

    df = get_df(source)

    # filter
    if filter_query is not None:
        filtering_expressions = filter_query.split(' && ')
        for filter_part in filtering_expressions:
            col_name, operator, filter_value = split_filter_part(filter_part)

            if col_name not in df.columns:
                continue

            if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
                df = df.loc[getattr(df[col_name], operator)(filter_value)]
            elif operator == 'contains':
                df = df[~df[col_name].isna()]
                df = df.loc[df[col_name].str.contains(filter_value)]
            elif operator == 'datestartswith':
                df = df.loc[df[col_name].str.startswith(filter_value)]
    # sort
    if sort_by is not None and len(sort_by) > 0:
        df = df.sort_values([col['column_id'] for col in sort_by],
                            ascending=[col['direction'] == 'asc' for col in sort_by],
                            inplace=False)

    if source_dict[source] == 'showers':
        df.__class__ = ShowerDataFrame
        return df
    else:
        df.__class__ = BolideDataFrame
        df = df.filter_date(start=start_date, end=end_date)
        if len(boundary_checklist) > 0 and 'Filter by FOV' in filter_fov:
            intersection = "Intersection" in filter_fov
            df = df.filter_boundary(boundary_checklist, intersection=intersection)

        del df['geometry']
        return df


def get_df_from_idx(source, rows):
    if source is None:
        df = pd.DataFrame({})

    else:
        df = get_df(source)
        df = df.loc[rows]

    if 'geometry' in df.columns:
        del df['geometry']

    if source is not None and source_dict[source] == 'showers':
        df.__class__ = ShowerDataFrame
    else:
        df.__class__ = BolideDataFrame
    return df


app.layout = html.Div(style={}, children=[
    html.Div(children=[
    html.H1(
        children='🌠 Interactive Bolide Data Visualizer',
        style={'textAlign': 'center'}),

    dcc.Markdown('''
------
This is a prototype webapp for interactively visualizing bolide detections from two sources:
- US Government satellites. See [here](https://cneos.jpl.nasa.gov/fireballs/) for more details.
- Human-vetted bolide detections from the Geostationary Lightning Mapper (GLM) instruments aboard
GOES-16 and GOES-17. See [here](https://neo-bolide.ndc.nasa.gov/) for more details.

This webapp uses the [`bolides`](https://bolides.readthedocs.io) Python package as a backend.
The package is recommended for more thorough data analysis.
The export button above the table will export filtered data to a .csv file that
can be read by `bolides` or any spreadsheet software.

------
To get started, select a data source below. If you have GLM data selected,
you may also click on a bolide detection on the map to plot its (uncalibrated)
light curve.
'''
    ),

    dcc.Dropdown(list(source_dict.keys()), id='source-select', placeholder='Select a data source'),

    dcc.Dropdown(id='color', placeholder='Select a variable to color by',
                 style={'width': '50vw', 'display': 'inline-block'}),

    dcc.Checklist(['log color scale'], [], inline=True, id='log-color',
                  style={'display': 'inline-block'}),

    html.Div(children=['Date filters: ',
             dcc.Input(type='text', id='date-start', placeholder='start date yyyy-mm-dd',
                       style={'display': 'inline-block'}, debounce=True),
             dcc.Input(type='text', id='date-end', placeholder='end date yyyy-mm-dd',
                       style={'display': 'inline-block'}, debounce=True)]),

    html.Div(children=['Field-of-view options:  ',
        dcc.Checklist(['goes', 'goes-e', 'goes-w', 'goes-w-ni', 'goes-w-i',
                       'fy4a', 'fy4a-n', 'fy4a-s', 'gmn-25km', 'gmn-70km', 'gmn-100km'],
                      [], inline=True, id='boundary-checklist',
                      style={'display': 'inline-block'})]),

    dcc.Checklist(['Filter by FOV', 'Intersection'], [], id='filter-fov'),

    html.Div(children=["Map projection: ",
                       dcc.Dropdown(projections, value='eckert4', id='projection',
                                    style={'width': '50vw', 'display': 'inline-block'})]),

    dcc.Graph(id='main-map', config={'displaylogo': False}),

    html.Div(children=[
        html.Div(children=[
            dcc.Dropdown([], id='scatter-x',
                placeholder='Select an x-variable',
                style={'display': 'inline-block', 'width': '20vw', 'margin': '0 auto'}),
            dcc.Dropdown([], id='scatter-y',
                placeholder='Select a y-variable',
                style={'display': 'inline-block', 'width': '20vw', 'margin': '0 auto'}),
            dcc.Graph(id='scatter', config={'displaylogo': False})],
            style={'display': 'inline-block', 'width': '40vw', 'margin': '0 auto'}),
        html.Div(children=[
            dcc.Dropdown(['latitude'], 'latitude', id='hist-var',
                         placeholder='Select a variable to make a histogram from'),
            html.Div(children=["Number of bins: ",
                dcc.Input(type='number', min=1, max=500, value=30, id='hist-bins', style={'display': 'inline-block'})]),
            dcc.Graph(id='hist', config={'displaylogo': False})],
            style={'display': 'inline-block', 'width': '40vw', 'margin': '0 auto'})
        ]),
    html.Div(children=[], id='lightcurve'),
    html.Button("Export light curve(s) as csv", id="save-button-lc", style={'display': 'none'}),
    dcc.Download(id='download-lc'),
    dcc.Store(id='lc-store'),
    html.A(' More data on this bolide', href="", id='bolide-link'),
    ], style={'width': '90vw', 'margin': '0 auto'}),

    dcc.Download(id="download"),
    html.Div(children=['You may type filter queries (including using operators like <, =, and >) in the table below. ',
                       html.Button("Export filtered data as csv",
                       id="save-button", style={'display': 'inline-block'})]),
    dash_table.DataTable(data=None,
        id='main-table',
        columns=[],
        filter_action="custom",
        sort_action="custom",
        sort_mode="single",
        column_selectable=False,
        row_selectable=False,
        row_deletable=False,
        selected_columns=[],
        selected_rows=[],
        page_action="custom",
        page_current=0,
        page_size=50,
        style_table={'overflowX': 'scroll'},
        style_cell={'font-size': '12px'}
    ),

    dcc.Store(id='data-rows')

])


@app.callback(
Output('main-table', 'data'),
Output('main-table', 'columns'),
Output('main-table', 'page_count'),
Output('data-rows', 'data'),
Input('source-select', 'value'),
Input('main-table', "page_current"),
Input('main-table', "page_size"),
Input('main-table', "filter_query"),
Input('date-start', 'value'),
Input('date-end', 'value'),
Input('filter-fov', 'value'),
Input('boundary-checklist', 'value'),
Input('main-table', 'sort_by')
)
def update_data(source, page_current, page_size, filter_query, start_date, end_date, filter_fov, boundary_checklist, sort_by):
    print('updating table')

    start_date = validate_iso(start_date)
    end_date = validate_iso(end_date)

    df = get_df_from_filters(source, filter_query, start_date, end_date,
                             filter_fov, boundary_checklist, sort_by)
    page_count = len(df)//page_size + 1
    data = df.iloc[page_current * page_size:(page_current + 1) * page_size].to_dict('records')
    columns = [{"name": i, "id": i, "deletable": True, "selectable": True} for i in df.columns]

    numeric_columns = []
    import numbers
    if len(df) > 0:
        for col in df.columns:
            if all(isinstance(x, numbers.Number) for x in df[col]):
                numeric_columns.append(col)

        from dash.dash_table.Format import Format
        for col in columns:
            if col['name'] in numeric_columns:
                col['type'] = 'numeric'
                col['format'] = Format(precision=3)

    rows = list(df.index)
    return data, columns, page_count, rows


def validate_iso(datestr):
    if datestr is not None:
        try:
            dt = datetime.datetime.fromisoformat(datestr)
        except ValueError:
            datestr = None
    return datestr


@app.callback(
Output('scatter-x', 'options'),
Output('scatter-y', 'options'),
Output('hist-var', 'options'),
Output('color', 'options'),
Output('scatter-x', 'value'),
Output('scatter-y', 'value'),
Output('hist-var', 'value'),
Output('color', 'value'),
Input('source-select', 'value'),
State('scatter-x', 'value'),
State('scatter-y', 'value'),
State('hist-var', 'value'),
State('color', 'value')
)
def update_dropdowns(source, scatter_x, scatter_y, hist_var, color):
    if source is None:
        return [], [], [], [], None, None, None, None
    df = get_df(source)

    numeric_columns = []
    color_columns = []
    import numbers
    if len(df) > 0:
        for col in df.columns:
            if all(isinstance(x, numbers.Number) for x in df[col]):
                numeric_columns.append(col)
                color_columns.append(col)
            elif len(df[col].unique()) < 20 and len(df[col].unique()) > 1:
                color_columns.append(col)

        if source_dict[source] != 'showers':
            numeric_columns.insert(0, 'datetime')

    if scatter_x not in numeric_columns:
        scatter_x = None
    if scatter_y not in numeric_columns:
        scatter_y = None
    if hist_var not in numeric_columns:
        hist_var = None
    if color not in color_columns:
        color = None

    return numeric_columns, numeric_columns, numeric_columns, color_columns, scatter_x, scatter_y, hist_var, color


@app.callback(
Output('scatter', 'figure'),
Input('scatter-x', 'value'),
Input('scatter-y', 'value'),
Input('color', 'value'),
Input('log-color', 'value'),
Input('data-rows', 'data'),
Input('source-select', 'value')
)
def update_scatter(x, y, color_column, log_color, rows, source):
    print('updating scatter')
    df = get_df_from_idx(source, rows)
    fig = go.Figure()
    if len(df) > 0 and x is not None and y is not None:
        import numbers
        if 'log color scale' in log_color and all([isinstance(x, numbers.Number) for x in df[color_column]]):
            fig = px.scatter(df, x=x, y=y, color=np.log(df[color_column]))
            fig.update_layout(coloraxis_colorbar={'title': 'log('+color_column+')'})
        else:
            fig = px.scatter(df, x=x, y=y, color=color_column)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig


@app.callback(
Output('hist', 'figure'),
Input('hist-var', 'value'),
Input('hist-bins', 'value'),
Input('source-select', 'value'),
Input('data-rows', 'data'),
Input('color', 'value')
)
def update_hist(var, bins, source, rows, color_column):
    print('updating histogram')
    if bins is None:
        bins = 1
    df = get_df_from_idx(source, rows)
    fig = go.Figure()
    if len(df) > 0 and var is not None:
        start = min(df[var])
        end = max(df[var])
        size = (end-start)/bins
        xbins = dict(start=start, end=end, size=size)
        if color_column is not None and len(df[color_column].unique()) < 20:
            for cat in df[color_column].unique():
                fig.add_trace(go.Histogram(x=df[df[color_column] == cat][var], name=cat))
        else:
            fig.add_trace(go.Histogram(x=df[var]))
        fig.update_traces(xbins=xbins)
        fig.update_layout(barmode='stack')
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, hovermode='x unified')
    return fig


@app.callback(
Output("download", "data"),
Input("save-button", "n_clicks"),
State('source-select', 'value'),
State('data-rows', 'data'))
def download_as_csv(n_clicks, source, rows):
    df = get_df_from_idx(source, rows)
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
Input("source-select", "value"),
Input("data-rows", "data"),
Input("color", "value"),
Input('log-color', 'value'),
Input('boundary-checklist', 'value'),
Input('projection', 'value'),
Input('main-table', "page_current"),
Input('main-table', "page_size"))
def update_map(source, rows, color_column, log_color, boundary_checklist, projection, page_current, page_size):
    print('updating map')
    df = get_df_from_idx(source, rows)
    if color_column not in df.columns:
        color_column = None

    logscale = 'log color scale' in log_color
    if projection == 'eckert4 ':
        projection = 'eckert4'
    if source is not None and source_dict[source] == 'showers':
        df = df.iloc[page_current * page_size:(page_current + 1) * page_size]
        df.__class__ = ShowerDataFrame
        fig = df.plot_orbits(use_3d=True)._figure
    else:
        fig = df.plot_interactive(projection, boundary_checklist, color_column, logscale)
    fig.update_layout(uirevision=str(source)+str(projection))
    return fig


@app.callback(
Output('lightcurve', 'children'),
Output('lc-store', 'data'),
Output('save-button-lc', 'style'),
Output('bolide-link', 'style'),
Output('bolide-link', 'href'),
Input('main-map', 'clickData'),
State("source-select", "value"),
State("data-rows", "data"),
State("color", "value")
)
def update_lightcurve(clickData, source, rows, color_column):
    if source is None or source_dict[source] != 'website':
        return [], [], {'display': 'none'}, {'display': 'none'}, ""
    print('updating lightcurve')
    df = get_df_from_idx(source, rows)
    if len(df) == 0:
        return [], [], {'display': 'none'}, {'display': 'none'}, ""
    fig = go.Figure()
    csvs = []
    cat_idx = clickData['points'][0]['curveNumber']
    if cat_idx != 0:
        cat = df[color_column].unique()[cat_idx]
        df = df[df[color_column] == cat]
    df = df.iloc[clickData['points'][0]['pointIndex']]

    df = pd.DataFrame(df).T
    df.to_csv('tmp.csv')
    bdf = BolideDataFrame('csv', 'tmp.csv')
    bdf.add_website_data()
    lcc = bdf.lightcurves.iloc[0]
    _id = bdf._id.iloc[0]
    dfs = []
    for lc in lcc:
        time = [datetime.datetime.utcfromtimestamp(t) for t in lc.time.value]
        lc_df = pd.DataFrame({'time': time, 'flux': lc.flux.value, 'source': lc.meta['LABEL']})
        dfs.append(lc_df)
        lc_df = lc_df.copy()
        sat = lc_df['source'][0]
        lc_df['time_'+sat] = lc_df['time']
        lc_df['flux_'+sat] = lc_df['flux']
        del lc_df['source'], lc_df['time'], lc_df['flux']
        csvs.append(lc_df.to_csv())
    lcc_df = pd.concat(dfs)
    fig = px.line(lcc_df, x='time', y='flux', color='source',
                  title='Light curve for bolide with ID '+_id)
    fig.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0}, hovermode='x unified')

    bolide_link = "https://neo-bolide.ndc.nasa.gov/#/eventdetail/"+_id

    return [dcc.Graph(figure=fig, config={'displaylogo': False})], [_id]+csvs, {}, {}, bolide_link


@app.callback(
Output("download-lc", "data"),
Input("save-button-lc", "n_clicks"),
State('lc-store', 'data'))
def download_lc(n_clicks, data):
    if not n_clicks:
        raise PreventUpdate
    _id = data[0]
    del data[0]
    dfs = [pd.read_csv(io.StringIO(csv)) for csv in data]
    df = pd.concat(dfs, axis=1, ignore_index=False)
    download_buffer = io.StringIO()
    df.to_csv(download_buffer, index=False)
    download_buffer.seek(0)
    filename = _id+'_light_curve.csv'
    return dict(content=download_buffer.getvalue(), filename=filename)


app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        <link rel="icon" type="image/svg" href="/assets/favicon.svg">
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''
if __name__ == '__main__':
    app.run_server(debug=True)
