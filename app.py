
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Spotify Spain Top 50",
    page_icon="🎵",
    layout="wide"
)

st.title("Spotify Spain Top 50")
st.subheader("Chart Lifecycle Analysis & Machine Learning")

# Load the cleaned dataset
file_path = Path("clean_data.csv")

if not file_path.exists():
    st.error("Please place clean_data.csv in the project folder.")
    st.stop()

df = pd.read_csv(file_path)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Sidebar filters
st.sidebar.header("Explore the chart")

min_date = df["date"].min().date()
max_date = df["date"].max().date()

date_range = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    df = df[
        df["date"].between(
            pd.Timestamp(date_range[0]),
            pd.Timestamp(date_range[1])
        )
    ]

if "artist" in df.columns:
    artists = sorted(df["artist"].dropna().unique())
    selected_artists = st.sidebar.multiselect("Filter artists", artists)

    if selected_artists:
        df = df[df["artist"].isin(selected_artists)]

# Summary metrics
c1, c2, c3 = st.columns(3)
c1.metric("Chart entries", f"{len(df):,}")
c2.metric("Unique songs", f"{df['song'].nunique():,}")
c3.metric("Unique artists", f"{df['artist'].nunique():,}")

# Song ranking chart
st.header("Song chart performance")

songs = sorted(df["song"].dropna().unique())

if songs:
    song = st.selectbox("Choose a song", songs)
    song_data = df[df["song"] == song].sort_values("date")

    st.line_chart(
        song_data.set_index("date")["position"]
    )

    st.caption(
        "A lower position means a better rank: position 1 is #1."
    )

# Artist analysis
st.header("Artist chart appearances")

artist_counts = df["artist"].value_counts().head(15)
st.bar_chart(artist_counts)

# Data preview and download
with st.expander("View chart data"):
    st.dataframe(df, use_container_width=True)

st.download_button(
    "Download filtered data",
    df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_chart_data.csv",
    mime="text/csv"
)

st.caption(
    "Educational project using historical chart data. "
    "Not an official Spotify product."
)
