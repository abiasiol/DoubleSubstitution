import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
import plotly.express as px
import pandas as pd
import base64
import datetime
import io
import double_substitution as ds
from dash.dependencies import Input, Output, State
from dash import dash_table

external_stylesheets = [dbc.themes.LUX]
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)
app.title = 'Andrea 2xSubstitution'

server = app.server

app.layout = html.Div(
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Double Substitution Fix"), className="mt-4")
        ]),
        dbc.Row([
            dbc.Col(html.H6(
                children='by Andrea Biasioli'),
                className="mb-4")
        ]),
        html.Br(),
        dbc.Row([
            dbc.Col(
                    dcc.Upload(
                        id='upload-data',
                        accept='.dvw',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files')
                        ]),
                        style={
                            'width': '100%',
                            'height': '450px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '10px',
                            'textAlign': 'center',
                            'margin': '10px',
                        },
                        # Allow multiple files to be uploaded
                        multiple=False
                    ), lg=6, md=6, xs=12, style={'padding': 35}
            ),
            dbc.Col(
                    html.Ol([
                                html.Li('Make sure that the setters of each team are marked with "5" in the scout file (see image below)'),
                                html.Li('Drag and drop / select the dvw scout file'),
                                html.Li('You can now download the fixed file!'),
                                html.Br(),
                                html.Img(src="assets/setters.png", height="350px"),
                            ]), lg=6, md=6, xs=12, style={'padding': 35}
            ),
        ]),



        html.Br(),
        dcc.Download(id="download-dvw"),
        dcc.Store(id='file-output'),

        html.Div(id='output-data-upload'),

        dbc.Row([
            dbc.Col(
                dash_table.DataTable(
                    id='table-log',
                    columns=[
                        dict(name='Timecode', id='timecode'),
                        dict(name='Code', id='code'),
                        dict(name='Rally', id='rally'),
                        dict(name='Notes', id='notes'),
                        dict(name='Team', id='team')
                    ],
                    data=pd.DataFrame(columns=['timecode', 'code', 'rally', 'notes', 'team']).to_dict('records'),
                    style_data={
                        'whiteSpace': 'normal',
                        'height': 'auto',
                        'lineHeight': '22px'
                    },
                    style_cell={'fontSize': 12},
                    style_data_conditional=[
                        {
                            'if': {
                                'filter_query': '{team} contains "home"'
                            },
                            'backgroundColor': 'rgba(188,212,230,0.2)',
                        },
                        {
                            'if': {
                                'filter_query': '{team} contains "guest"'
                            },
                            'backgroundColor': 'rgba(250,128,114, 0.15)',
                        },

                    ]
                ), lg=12, md=12, xs=12, style={'padding': 35}),
        ]),

    ])
)


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        if '.dvw' in filename:
            # Assume that the user uploaded a CSV file
            log, file_out = ds.perform_double_substitution(io.StringIO(decoded.decode('utf-8')))

            df_log = pd.read_csv(io.StringIO(log), sep=';', header=None,
                                 names=['timecode', 'code', 'rally', 'notes', 'team'])
            df_log = df_log.sort_values(by='timecode', ascending=True)

            return html.Div(['Done!']), df_log.to_dict('records'), dict(content=file_out.getvalue(),
                                                                        filename=f'{filename[:-4]}_new.dvw')
        else:
            return html.Div(['Nononono']), pd.DataFrame(columns=['timecode', 'code', 'rally', 'notes', 'team']).to_dict(
                'records'), None

    except Exception as e:
        print(e)
        return html.Div(['There was an error processing this file.']), pd.DataFrame(
            columns=['timecode', 'code', 'rally', 'notes', 'team']).to_dict('records'), None


@app.callback([Output('output-data-upload', 'children'),
               Output('table-log', 'data'),
               Output('file-output', 'data')],
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(content, name, date):
    if content is not None:
        children, log, download_data = parse_contents(content, name, date)
        return children, log, download_data
    else:
        return html.Div(['']), pd.DataFrame(columns=['timecode', 'code', 'rally', 'notes', 'team']).to_dict(
            'records'), None


@app.callback(
    Output('download-dvw', 'data'),
    Input('file-output', 'data')
)
def send_file_out(data):
    if data:
        return data


# @app.callback(
#     Output("download-text", "data"),
#     Input("btn-download-txt", "n_clicks"),
#     prevent_initial_call=True,
# )
# def func(n_clicks):
#     return dict(content="Hello world!", filename="hello.txt")


if __name__ == '__main__':
    app.run_server(debug=True)
