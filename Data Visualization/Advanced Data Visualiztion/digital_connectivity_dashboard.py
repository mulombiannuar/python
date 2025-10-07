"""
Advanced Visualization Dashboard (Dash + Plotly)
Single-file Dash app that:
 - loads the excel dataset
 - cleans the data
 - shows an interactive choropleth, bar chart, scatter, network graph, and data table
 - provides simple dropdowns/controls for interactivity

Requirements:
 pip install pandas dash plotly networkx pycountry

Run:
 python app.py
 Then open http://127.0.0.1:8050 in your browser.
"""

import pandas as pd
import numpy as np
import dash  # Main Dash framework for building web apps
from dash import (
    dcc,
    html,
    dash_table,
)  # Import core Dash components: graphs, HTML, and tables
from dash.dependencies import (
    Input,
    Output,
    State,
)  # For linking UI elements via callbacks
import plotly.express as px  # Simplified interface for creating interactive plots
import plotly.graph_objects as go  # Low-level API for building custom Plotly visualizations
import networkx as nx  # Library for creating and analyzing network graphs
import pycountry  # Library for working with standardized country names and ISO codes


# --- 1) Load data from Excel
df = pd.read_excel(
    "School-Age-Digital-Connectivity Dataset.xlsx", sheet_name="Total school age"
)

# --- 2) Standardize column names to the expected schema in your prompt
# The sheet contains slightly different headings; map to expected names.
expected_cols = [
    "ISO3",
    "Countries and areas",
    "Region",
    "Sub-region",
    "Income Group",
    "Total",
    "Rural",
    "Urban",
    "Poorest",
    "Richest",
    "Data source",
    "Time period",
]

# We will do a best-effort mapping based on common header names observed in the sheet.
col_map_candidates = {
    # Possible names from the sheet -> desired name
    "ISO3": "ISO3",
    "Countries and areas": "Countries and areas",
    "Region": "Region",
    "Sub-region": "Sub-region",
    "Income Group": "Income Group",
    "Total": "Total",
    "Residence": None,  # sometimes there's combined heading
    "Rural": "Rural",
    "Urban": "Urban",
    "Wealth quintile": None,
    "Poorest": "Poorest",
    "Richest": "Richest",
    "Source": "Data source",
    "Data source": "Data source",
    "Time period": "Time period",
}

# Normalize and map whatever columns we have
current_cols = df.columns.tolist()
mapped = {}
for c in current_cols:
    if c in col_map_candidates and col_map_candidates[c]:
        mapped[c] = col_map_candidates[c]
    else:
        # Some sheets have slightly different header strings (trim and match heuristically)
        c_trim = c.strip().lower()
        if "iso" in c_trim:
            mapped[c] = "ISO3"
        elif "country" in c_trim or "countries" in c_trim or "area" in c_trim:
            mapped[c] = "Countries and areas"
        elif "region" == c_trim or "sub-region" in c_trim or "sub region" in c_trim:
            if "sub" in c_trim:
                mapped[c] = "Sub-region"
            else:
                mapped[c] = "Region"
        elif "income" in c_trim:
            mapped[c] = "Income Group"
        elif "total" in c_trim and "total" not in mapped.values():
            mapped[c] = "Total"
        elif "rural" in c_trim:
            mapped[c] = "Rural"
        elif "urban" in c_trim:
            mapped[c] = "Urban"
        elif "poorest" in c_trim:
            mapped[c] = "Poorest"
        elif "richest" in c_trim:
            mapped[c] = "Richest"
        elif "source" in c_trim:
            mapped[c] = "Data source"
        elif "time" in c_trim:
            mapped[c] = "Time period"
        else:
            # if unknown, keep column (will drop later if unused)
            mapped[c] = c

df = df.rename(columns=mapped)

# Ensure all expected columns exist; create if missing (empty)
for c in expected_cols:
    if c not in df.columns:
        df[c] = np.nan

# --- 3) Clean numeric columns: remove % and convert to numeric (0-100)
pct_columns = ["Total", "Rural", "Urban", "Poorest", "Richest"]


def parse_pct(x):
    if pd.isna(x):
        return np.nan
    # remove stray characters and convert to numeric
    s = str(x).replace("%", "").replace(",", "").strip()
    try:
        return float(s)
    except:
        # some cells contain combined text; attempt to extract leading number
        import re

        m = re.search(r"(\d+(\.\d+)?)", s)
        if m:
            return float(m.group(1))
        return np.nan


for c in pct_columns:
    if c in df.columns:
        df[c] = df[c].apply(parse_pct)

# Drop rows with no ISO3 (these are unlikely to map)
df = df[df["ISO3"].notna()].copy()

# Some ISO codes might be like 'BOL' vs 'BOLivia' etc; ensure ISO3 are uppercase 3-letter
df["ISO3"] = df["ISO3"].str.strip().str.upper().str[:3]

# Provide a clean display name column
df["Country"] = df["Countries and areas"].fillna(
    df["Country"] if "Country" in df.columns else np.nan
)

# --- 4) Fill missing regions/income group with 'Unknown' to avoid empty filters
df["Region"] = df["Region"].fillna("Unknown")
df["Sub-region"] = df["Sub-region"].fillna("Unknown")
df["Income Group"] = df["Income Group"].fillna("Unknown")
df["Time period"] = df["Time period"].fillna("Unknown")

# --- 4.1) Calculate gaps (from code 2)
df["Urban_Rural_Gap"] = df["Urban"] - df["Rural"]
df["Wealth_Gap"] = df["Richest"] - df["Poorest"]


# --- 5) Helper: map ISO3 to full country names (if Country column is empty)
def iso3_to_name(iso3):
    try:
        c = pycountry.countries.get(alpha_3=iso3)
        return c.name
    except:
        return iso3


df["Country"] = df["Country"].fillna(df["ISO3"].apply(iso3_to_name))

# --- 6) Build aggregated summaries for visuals
regions = sorted(df["Region"].unique())
income_groups = sorted(df["Income Group"].unique())

# Default metric for visualization - extended with gap metrics
METRICS = [
    "Total",
    "Rural",
    "Urban",
    "Poorest",
    "Richest",
    "Urban_Rural_Gap",
    "Wealth_Gap",
]

# Enhanced color scheme from code 2
colors = {
    "background": "#ffffff",
    "card": "#f8f9fa",
    "card_border": "#dee2e6",
    "text": "#212529",
    "text_muted": "#6c757d",
    "primary": "#0d6efd",
    "secondary": "#6610f2",
    "success": "#198754",
    "warning": "#ffc107",
    "danger": "#dc3545",
    "info": "#0dcaf0",
    "dark": "#212529",
    "light": "#f8f9fa",
}

# --- 7) Create Dash app with callback exceptions suppressed
app = dash.Dash(
    __name__,
    title="Advanced Visualisation Dashboard",
    suppress_callback_exceptions=True,
)
server = app.server

# Enhanced app layout with styling from code 2
app.layout = html.Div(
    [
        # Enhanced header from code 2
        html.Div(
            [
                html.H1(
                    ["School-Age Digital Connectivity Dashboard"],
                    style={
                        "color": colors["primary"],
                        "fontWeight": "700",
                        "fontSize": "2.5rem",
                        "textAlign": "center",
                        "marginBottom": "10px",
                        "textShadow": "2px 2px 4px rgba(0,0,0,0.1)",
                    },
                ),
                html.P(
                    "Comprehensive global analysis of internet access among school-age children",
                    style={
                        "color": colors["text_muted"],
                        "fontSize": "1.2rem",
                        "textAlign": "center",
                        "marginBottom": "30px",
                    },
                ),
            ],
            style={
                "background": f'linear-gradient(135deg, {colors["light"]} 0%, {colors["card"]} 100%)',
                "padding": "30px",
                "borderRadius": "15px",
                "boxShadow": "0 4px 6px rgba(0,0,0,0.1)",
                "marginBottom": "20px",
            },
        ),
        # Key metrics cards from code 2
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    "📍",
                                    style={"fontSize": "2rem", "marginRight": "15px"},
                                ),
                                html.Div(
                                    [
                                        html.H6(
                                            "Total Countries",
                                            style={
                                                "color": colors["text_muted"],
                                                "marginBottom": "5px",
                                                "fontSize": "0.9rem",
                                            },
                                        ),
                                        html.H2(
                                            len(df),
                                            style={
                                                "color": colors["primary"],
                                                "fontWeight": "bold",
                                                "margin": "0",
                                            },
                                        ),
                                    ]
                                ),
                            ],
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "padding": "15px",
                            },
                        )
                    ],
                    style={
                        "backgroundColor": colors["card"],
                        "border": f'2px solid {colors["primary"]}',
                        "borderRadius": "10px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                        "margin": "5px",
                        "flex": "1",
                    },
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    "📈",
                                    style={"fontSize": "2rem", "marginRight": "15px"},
                                ),
                                html.Div(
                                    [
                                        html.H6(
                                            "Average Connectivity",
                                            style={
                                                "color": colors["text_muted"],
                                                "marginBottom": "5px",
                                                "fontSize": "0.9rem",
                                            },
                                        ),
                                        html.H2(
                                            f"{df['Total'].mean():.0f}%",
                                            style={
                                                "color": colors["success"],
                                                "fontWeight": "bold",
                                                "margin": "0",
                                            },
                                        ),
                                    ]
                                ),
                            ],
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "padding": "15px",
                            },
                        )
                    ],
                    style={
                        "backgroundColor": colors["card"],
                        "border": f'2px solid {colors["success"]}',
                        "borderRadius": "10px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                        "margin": "5px",
                        "flex": "1",
                    },
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    "🌍",
                                    style={"fontSize": "2rem", "marginRight": "15px"},
                                ),
                                html.Div(
                                    [
                                        html.H6(
                                            "Geographic Regions",
                                            style={
                                                "color": colors["text_muted"],
                                                "marginBottom": "5px",
                                                "fontSize": "0.9rem",
                                            },
                                        ),
                                        html.H2(
                                            df["Region"].nunique(),
                                            style={
                                                "color": colors["info"],
                                                "fontWeight": "bold",
                                                "margin": "0",
                                            },
                                        ),
                                    ]
                                ),
                            ],
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "padding": "15px",
                            },
                        )
                    ],
                    style={
                        "backgroundColor": colors["card"],
                        "border": f'2px solid {colors["info"]}',
                        "borderRadius": "10px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                        "margin": "5px",
                        "flex": "1",
                    },
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    "⚖️",
                                    style={"fontSize": "2rem", "marginRight": "15px"},
                                ),
                                html.Div(
                                    [
                                        html.H6(
                                            "Avg Wealth Gap",
                                            style={
                                                "color": colors["text_muted"],
                                                "marginBottom": "5px",
                                                "fontSize": "0.9rem",
                                            },
                                        ),
                                        html.H2(
                                            f"{df['Wealth_Gap'].mean():.0f}%",
                                            style={
                                                "color": colors["danger"],
                                                "fontWeight": "bold",
                                                "margin": "0",
                                            },
                                        ),
                                    ]
                                ),
                            ],
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "padding": "15px",
                            },
                        )
                    ],
                    style={
                        "backgroundColor": colors["card"],
                        "border": f'2px solid {colors["danger"]}',
                        "borderRadius": "10px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                        "margin": "5px",
                        "flex": "1",
                    },
                ),
            ],
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "marginBottom": "20px",
            },
        ),
        # Enhanced filters with styling from code 2
        html.Div(
            [
                html.Div(
                    [
                        html.Label(
                            "Select Region:",
                            style={
                                "fontWeight": "bold",
                                "marginBottom": "8px",
                                "color": colors["text"],
                                "fontSize": "1rem",
                            },
                        ),
                        dcc.Dropdown(
                            id="region-filter",
                            options=[{"label": "All Regions", "value": "All"}]
                            + [{"label": r, "value": r} for r in regions],
                            value="All",
                            style={
                                "borderRadius": "8px",
                                "border": f'2px solid {colors["card_border"]}',
                            },
                        ),
                    ],
                    style={
                        "width": "24%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                        "paddingRight": "1%",
                    },
                ),
                html.Div(
                    [
                        html.Label(
                            "Income Group:",
                            style={
                                "fontWeight": "bold",
                                "marginBottom": "8px",
                                "color": colors["text"],
                                "fontSize": "1rem",
                            },
                        ),
                        dcc.Dropdown(
                            id="income-filter",
                            options=[{"label": "All Income Groups", "value": "All"}]
                            + [{"label": ig, "value": ig} for ig in income_groups],
                            value="All",
                            style={
                                "borderRadius": "8px",
                                "border": f'2px solid {colors["card_border"]}',
                            },
                        ),
                    ],
                    style={
                        "width": "24%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                        "paddingRight": "1%",
                    },
                ),
                html.Div(
                    [
                        html.Label(
                            "View Type:",
                            style={
                                "fontWeight": "bold",
                                "marginBottom": "8px",
                                "color": colors["text"],
                                "fontSize": "1rem",
                            },
                        ),
                        dcc.Dropdown(
                            id="view-selector",
                            options=[
                                {"label": "Overview", "value": "overview"},
                                {"label": "Inequality Analysis", "value": "inequality"},
                                {"label": "Rankings", "value": "rankings"},
                            ],
                            value="overview",
                            style={
                                "borderRadius": "8px",
                                "border": f'2px solid {colors["card_border"]}',
                            },
                        ),
                    ],
                    style={
                        "width": "24%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                        "paddingRight": "1%",
                    },
                ),
                html.Div(
                    [
                        html.Label(
                            "Metric:",
                            style={
                                "fontWeight": "bold",
                                "marginBottom": "8px",
                                "color": colors["text"],
                                "fontSize": "1rem",
                            },
                        ),
                        dcc.Dropdown(
                            id="metric",
                            options=[{"label": m, "value": m} for m in METRICS],
                            value="Total",
                            style={
                                "borderRadius": "8px",
                                "border": f'2px solid {colors["card_border"]}',
                            },
                        ),
                    ],
                    style={
                        "width": "24%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                    },
                ),
            ],
            style={
                "padding": "20px",
                "backgroundColor": colors["card"],
                "borderRadius": "10px",
                "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                "marginBottom": "20px",
            },
        ),
        # Main content area - dynamic based on view selector
        html.Div(id="dashboard-content"),
        # Data table (always visible but styled)
        html.Div(
            [
                html.H4(
                    "Data Table (filtered)",
                    style={"color": colors["text"], "marginTop": "30px"},
                ),
                dash_table.DataTable(
                    id="table",
                    columns=[
                        {"name": c, "id": c}
                        for c in [
                            "ISO3",
                            "Country",
                            "Region",
                            "Sub-region",
                            "Income Group",
                        ]
                        + METRICS
                        + ["Data source", "Time period"]
                    ],
                    page_size=12,
                    sort_action="native",
                    filter_action="native",
                    style_table={"overflowX": "auto"},
                    style_header={
                        "backgroundColor": colors["primary"],
                        "color": "white",
                        "fontWeight": "bold",
                    },
                    style_cell={
                        "backgroundColor": colors["card"],
                        "color": colors["text"],
                        "border": f'1px solid {colors["card_border"]}',
                    },
                ),
            ]
        ),
        html.Div(style={"height": "30px"}),
    ],
    style={
        "maxWidth": "1200px",
        "margin": "auto",
        "fontFamily": "Arial, sans-serif",
        "backgroundColor": colors["background"],
        "minHeight": "100vh",
        "padding": "20px",
    },
)

# --- 8) Callbacks to update visuals


def filter_df(region_value, income_value):
    dff = df.copy()
    if region_value and region_value != "All":
        dff = dff[dff["Region"] == region_value]
    if income_value and income_value != "All":
        dff = dff[dff["Income Group"] == income_value]
    return dff


@app.callback(
    Output("dashboard-content", "children"),
    [
        Input("region-filter", "value"),
        Input("income-filter", "value"),
        Input("view-selector", "value"),
        Input("metric", "value"),
    ],
)
def update_dashboard_content(region_value, income_value, view, metric):
    dff = filter_df(region_value, income_value)

    if view == "overview":
        return html.Div(
            [
                # Choropleth and scatter row
                html.Div(
                    [
                        html.Div(
                            [
                                html.H4(
                                    "Global Connectivity Map",
                                    style={"color": colors["text"]},
                                ),
                                dcc.Graph(id="choropleth"),
                            ],
                            style={
                                "width": "65%",
                                "display": "inline-block",
                                "paddingRight": "1%",
                            },
                        ),
                        html.Div(
                            [
                                html.H4(
                                    "Rural vs Urban Analysis",
                                    style={"color": colors["text"]},
                                ),
                                dcc.Graph(id="scatter"),
                            ],
                            style={
                                "width": "34%",
                                "display": "inline-block",
                                "verticalAlign": "top",
                            },
                        ),
                    ]
                ),
                # Top Countries on its own row
                html.Div(
                    [
                        html.Div(
                            [
                                html.H4(
                                    f"Top Countries by {metric}",
                                    style={"color": colors["text"]},
                                ),
                                dcc.Graph(id="bar-chart"),
                            ],
                            style={"width": "100%"},
                        ),
                    ],
                    style={"marginTop": "30px"},
                ),
                # Regional Hierarchy on its own row
                html.Div(
                    [
                        html.Div(
                            [
                                html.H4(
                                    "Regional Hierarchy",
                                    style={"color": colors["text"]},
                                ),
                                dcc.Graph(
                                    id="network-graph", style={"height": "600px"}
                                ),
                            ],
                            style={"width": "100%"},
                        ),
                    ],
                    style={"marginTop": "30px"},
                ),
            ]
        )

    elif view == "inequality":
        # Inequality analysis from code 2
        top_gaps = dff.nlargest(20, "Wealth_Gap")

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=top_gaps["Country"],
                x=top_gaps["Wealth_Gap"],
                name="Wealth Gap",
                orientation="h",
                marker_color=colors["danger"],
                marker_line_color="rgba(0,0,0,0.2)",
                marker_line_width=1.5,
            )
        )
        fig.add_trace(
            go.Bar(
                y=top_gaps["Country"],
                x=top_gaps["Urban_Rural_Gap"],
                name="Urban-Rural Gap",
                orientation="h",
                marker_color=colors["warning"],
                marker_line_color="rgba(0,0,0,0.2)",
                marker_line_width=1.5,
            )
        )
        fig.update_layout(
            title={
                "text": "Digital Inequality Analysis (Top 20 Countries)",
                "font": {"size": 20, "color": colors["text"]},
            },
            barmode="group",
            template="plotly_white",
            paper_bgcolor=colors["card"],
            plot_bgcolor="white",
            font={"color": colors["text"]},
            height=600,
            xaxis_title="Gap (%)",
            yaxis_title="Country",
        )

        return html.Div([dcc.Graph(figure=fig)])

    elif view == "rankings":
        # Rankings view from code 2
        top10 = dff.nlargest(10, "Total")
        bottom10 = dff.nsmallest(10, "Total")

        fig1 = go.Figure(
            go.Bar(
                x=top10["Total"],
                y=top10["Country"],
                orientation="h",
                marker_color=colors["success"],
                marker_line_color="rgba(0,0,0,0.2)",
                marker_line_width=1.5,
                text=top10["Total"],
                textposition="auto",
                texttemplate="%{text}%",
            )
        )
        fig1.update_layout(
            title={
                "text": "Top 10 Countries by Connectivity",
                "font": {"size": 16, "color": colors["text"]},
            },
            template="plotly_white",
            paper_bgcolor=colors["card"],
            plot_bgcolor="white",
            font={"color": colors["text"]},
            height=400,
        )

        fig2 = go.Figure(
            go.Bar(
                x=bottom10["Total"],
                y=bottom10["Country"],
                orientation="h",
                marker_color=colors["danger"],
                marker_line_color="rgba(0,0,0,0.2)",
                marker_line_width=1.5,
                text=bottom10["Total"],
                textposition="auto",
                texttemplate="%{text}%",
            )
        )
        fig2.update_layout(
            title={
                "text": "Bottom 10 Countries by Connectivity",
                "font": {"size": 16, "color": colors["text"]},
            },
            template="plotly_white",
            paper_bgcolor=colors["card"],
            plot_bgcolor="white",
            font={"color": colors["text"]},
            height=400,
        )

        return html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [dcc.Graph(figure=fig1)],
                            style={
                                "width": "48%",
                                "display": "inline-block",
                                "paddingRight": "2%",
                            },
                        ),
                        html.Div(
                            [dcc.Graph(figure=fig2)],
                            style={"width": "48%", "display": "inline-block"},
                        ),
                    ]
                )
            ]
        )

    # Return empty div for other cases
    return html.Div()


# Callbacks for dynamically created components - they will only be called when components exist
@app.callback(
    Output("choropleth", "figure"),
    [
        Input("region-filter", "value"),
        Input("income-filter", "value"),
        Input("metric", "value"),
    ],
    [State("view-selector", "value")],
)
def update_choropleth(region_value, income_value, metric, current_view):
    # Only update if we're in overview view
    if current_view != "overview":
        return go.Figure()

    dff = filter_df(region_value, income_value)
    # Use ISO3 for mapping; Plotly expects standard ISO alpha-3
    fig = px.choropleth(
        dff,
        locations="ISO3",
        color=metric,
        hover_name="Country",
        color_continuous_scale="Viridis",
        range_color=(
            (0, 100) if metric not in ["Urban_Rural_Gap", "Wealth_Gap"] else None
        ),
        labels={metric: metric + " (%)"},
        title=f"{metric} coverage by country",
    )
    fig.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        paper_bgcolor=colors["card"],
        font={"color": colors["text"]},
        height=500,
    )
    return fig


@app.callback(
    Output("bar-chart", "figure"),
    [
        Input("region-filter", "value"),
        Input("income-filter", "value"),
        Input("metric", "value"),
    ],
    [State("view-selector", "value")],
)
def update_bar(region_value, income_value, metric, current_view):
    # Only update if we're in overview view
    if current_view != "overview":
        return go.Figure()

    dff = filter_df(region_value, income_value)
    dff = (
        dff[["Country", "ISO3", metric]]
        .dropna(subset=[metric])
        .sort_values(metric, ascending=False)
        .head(12)
    )
    fig = px.bar(
        dff,
        x=metric,
        y="Country",
        orientation="h",
        text=metric,
        title=f"Top countries by {metric}",
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        margin={"l": 160},
        paper_bgcolor=colors["card"],
        plot_bgcolor="white",
        font={"color": colors["text"]},
        height=500,
    )
    return fig


@app.callback(
    Output("scatter", "figure"),
    [Input("region-filter", "value"), Input("income-filter", "value")],
    [State("view-selector", "value")],
)
def update_scatter(region_value, income_value, current_view):
    # Only update if we're in overview view
    if current_view != "overview":
        return go.Figure()

    dff = filter_df(region_value, income_value)
    # Scatter: Rural vs Urban (where both available)
    d = dff.dropna(subset=["Rural", "Urban"])
    if d.shape[0] == 0:
        fig = go.Figure()
        fig.update_layout(
            title="No data to show for Rural vs Urban scatter",
            paper_bgcolor=colors["card"],
            font={"color": colors["text"]},
            height=500,
        )
        return fig
    fig = px.scatter(
        d,
        x="Rural",
        y="Urban",
        hover_name="Country",
        size=np.clip(d["Total"].fillna(0), 1, 100),
        labels={"Rural": "Rural (%)", "Urban": "Urban (%)"},
        title="Rural vs Urban coverage (bubble size = Total)",
        color_discrete_sequence=[colors["primary"]],
    )
    fig.update_layout(
        paper_bgcolor=colors["card"],
        plot_bgcolor="white",
        font={"color": colors["text"]},
        height=500,
    )
    return fig


@app.callback(
    Output("network-graph", "figure"),
    [Input("region-filter", "value"), Input("income-filter", "value")],
    [State("view-selector", "value")],
)
def update_network(region_value, income_value, current_view):
    # Only update if we're in overview view
    if current_view != "overview":
        return go.Figure()

    dff = filter_df(region_value, income_value)
    # We'll build a 3-layer network: Region -> Sub-region -> Country
    G = nx.Graph()
    # Add region nodes
    for reg in sorted(dff["Region"].unique()):
        G.add_node(reg, layer="region")
    # Add subregion nodes and edges to region
    for _, row in dff.groupby(["Region", "Sub-region"]).size().reset_index().iterrows():
        reg = row["Region"]
        sub = row["Sub-region"]
        if not G.has_node(sub):
            G.add_node(sub, layer="subregion")
        G.add_edge(reg, sub)
    # Add countries and connect to sub-region
    for _, row in dff.iterrows():
        country = row["Country"]
        sub = row["Sub-region"]
        if not G.has_node(country):
            G.add_node(country, layer="country")
        # ensure sub exists
        if not G.has_node(sub):
            G.add_node(sub, layer="subregion")
        G.add_edge(sub, country)

    # Positioning: use spring layout for readability
    pos = nx.spring_layout(G, seed=42, k=0.8)

    # Build plotly edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    # Build node trace
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        node_text.append(n)
        layer = G.nodes[n].get("layer", "country")
        if layer == "region":
            node_color.append(colors["primary"])
            node_size.append(30)
        elif layer == "subregion":
            node_color.append(colors["info"])
            node_size.append(18)
        else:
            node_color.append(colors["success"])
            node_size.append(8)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        textposition="top center",
        text=node_text,
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            line_width=1,
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Network: Region → Sub-region → Country",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor=colors["card"],
            font={"color": colors["text"]},
            height=600,
        ),
    )
    return fig


@app.callback(
    Output("table", "data"),
    [
        Input("region-filter", "value"),
        Input("income-filter", "value"),
        Input("metric", "value"),
    ],
)
def update_table(region_value, income_value, metric):
    dff = filter_df(region_value, income_value)
    # Keep the columns in the order defined in layout
    cols = [
        "ISO3",
        "Country",
        "Region",
        "Sub-region",
        "Income Group",
        "Total",
        "Rural",
        "Urban",
        "Poorest",
        "Richest",
        "Urban_Rural_Gap",
        "Wealth_Gap",
        "Data source",
        "Time period",
    ]
    for c in cols:
        if c not in dff.columns:
            dff[c] = ""
    return dff[cols].to_dict("records")


# --- 9) Run server
if __name__ == "__main__":
    app.run(debug=True)
