from dash import Dash, Input, Output, callback, dcc, html

app = Dash(__name__)

app.layout = html.Div(
    [
        dcc.Input(id="input-1", value="initial value", type="text"),
        html.Div(id="intermediate-value"),
        html.Div(id="output"),
    ]
)


@app.callback(Output("intermediate-value", "children"), Input("input-1", "value"))
def callback_a(user_input):
    # some transformation to the input
    transformed_input = user_input + " has been transformed"
    return transformed_input


@app.callback(Output("output", "children"), Input("intermediate-value", "children"))
def callback_b(transformed_input):
    # use the transformed input from callback_a to produce output
    output = transformed_input + " and used in another callback"
    return output


if __name__ == "__main__":
    app.run_server(debug=True)
