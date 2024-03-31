import plotly.graph_objects as go
import torch


def plot():
    fig = go.Figure()
    fig.add_traces(
        go.Scatter(
            x=torch.linspace(0, 5, 5),
            y=torch.linspace(-3, 2, 6)
        )
    )
    fig.update_layout(
        # ['ggplot2', 'seaborn', 'simple_white', 'plotly', 'plotly_white', 'plotly_dark', 'presentation', 'xgridoff', 'ygridoff', 'gridon', 'none']
        template="simple_white",
    )
    fig.show()


if __name__ == "__main__":
    plot()
