"""Dynamic Power-BI-style chart builder: auto-suggests, accepts X/Y/type."""
import plotly.express as px
import pandas as pd


CHART_TYPES = ["bar", "line", "area", "scatter", "pie", "histogram", "box"]


def build(df: pd.DataFrame, chart_type: str, x: str, y: str = None, color: str = None, agg: str = "sum"):
    if df is None or df.empty or x not in df.columns:
        return None
    title = f"{chart_type.title()} of {y or 'count'} by {x}"
    try:
        if chart_type == "histogram":
            return px.histogram(df, x=x, color=color, title=title)
        if chart_type == "pie":
            if y and y in df.columns:
                grp = df.groupby(x, dropna=False)[y].agg(agg).reset_index()
                return px.pie(grp, names=x, values=y, title=title, hole=0.35)
            grp = df[x].value_counts().reset_index()
            grp.columns = [x, "count"]
            return px.pie(grp, names=x, values="count", title=f"Count of {x}", hole=0.35)

        if y is None:
            grp = df[x].value_counts().reset_index()
            grp.columns = [x, "count"]
            y_use = "count"
            data = grp
        else:
            if pd.api.types.is_numeric_dtype(df[y]):
                grp = df.groupby(x, dropna=False)[y].agg(agg).reset_index()
            else:
                grp = df.groupby(x, dropna=False)[y].count().reset_index()
            data = grp
            y_use = y

        if chart_type == "bar":
            return px.bar(data, x=x, y=y_use, color=color if color in data.columns else None, title=title)
        if chart_type == "line":
            return px.line(data, x=x, y=y_use, markers=True, title=title)
        if chart_type == "area":
            return px.area(data, x=x, y=y_use, title=title)
        if chart_type == "scatter":
            return px.scatter(df, x=x, y=y_use, color=color if color in df.columns else None, title=title)
        if chart_type == "box":
            return px.box(df, x=x, y=y_use, color=color if color in df.columns else None, title=title)
    except Exception:
        return None
    return None
