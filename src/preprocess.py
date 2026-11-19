"""
STEP 1: CLEAN THE RAW DATA
--------------------------
This file handles messy real-world data:
- Excel formula-wrapped dates (="2025-08-01...")
- JSON strings inside CSV cells
- Invalid coordinates (0,0)
- Missing values
"""

import json
import re
import pandas as pd
import numpy as np


def _parse_excel_datetime(s):
    """
    Handles Excel-exported dates that look like: ="2025-08-01 15:10:55.764952"
    The =" and " wrapping is an Excel artifact we need to strip.
    """
    if pd.isna(s):
        return pd.NaT  # "Not a Time" = pandas way of saying "invalid date"
    
    s = str(s).strip()
    
    # Remove Excel formula wrapping: ="..." or just "..."
    s = re.sub(r'^="|"$', '', s)
    
    # Try to convert to datetime
    return pd.to_datetime(s, errors='coerce')


def _json_field(x, key, default=np.nan):
    """
    AIS data stores SOG/COG inside a JSON string in the 'data' column.
    This safely extracts a field from that JSON.
    """
    try:
        # If x is a string, parse it as JSON. If already a dict, use directly.
        d = json.loads(x) if isinstance(x, str) else {}
        return d.get(key, default)
    except Exception:
        # If JSON is malformed, return the default
        return default


def _add_time_features(df, time_col):
    """
    Converts timestamp into "seconds since midnight" (0-86400).
    This helps the model understand time-of-day patterns.
    Example: 15:30:00 → 55800 seconds
    """
    dt = df[time_col]
    seconds = (dt.dt.hour * 3600 + 
               dt.dt.minute * 60 + 
               dt.dt.second + 
               dt.dt.microsecond / 1e6)
    df['time_seconds'] = seconds
    return df


def _course_features(df, col):
    """
    THE MAGIC FIX FOR COG:
    Converts 0-360 degree course into sin/cos pair.
    This makes 359° and 1° numerically close, as they should be.
    """
    # Convert degrees to radians, fill missing with 0
    rad = np.deg2rad(df[col].fillna(0).astype(float))
    
    # Add two new columns
    df[f'{col}_sin'] = np.sin(rad)
    df[f'{col}_cos'] = np.cos(rad)
    
    return df


def clean_ais(path):
    """
    Cleans AIS CSV file. Expected columns:
    message_type, mmsi, longitude, latitude, data, source_message, date_time
    """
    print(f"Loading AIS from: {path}")
    df = pd.read_csv(path)
    print(f"  Raw rows: {len(df)}")
    
    # Parse the messy Excel date format
    df['date_time'] = df['date_time'].apply(_parse_excel_datetime)
    
    # Extract SOG/COG from JSON 'data' column if not present as separate columns
    if 'sog' not in df.columns:
        print("  Extracting SOG from JSON data column...")
        df['sog'] = df['data'].apply(lambda x: _json_field(x, 'sog', 0.0))
    
    if 'cog' not in df.columns:
        print("  Extracting COG from JSON data column...")
        df['cog'] = df['data'].apply(lambda x: _json_field(x, 'cog', 0.0))
    
    # Keep only the columns we need
    keep = ['mmsi', 'longitude', 'latitude', 'sog', 'cog', 'date_time']
    df = df[keep].copy()
    
    # Convert to numeric, invalid values become NaN
    for c in ['longitude', 'latitude', 'sog', 'cog']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    # FILTER 1: Geographic bounds (Busan area)
    # This removes AIS type-5 static reports with (0,0) coordinates
    # and any global positioning errors
    df = df[(df['longitude'].between(128.5, 130.0)) & 
            (df['latitude'].between(34.5, 36.0))]
    print(f"  After geographic filter: {len(df)}")
    
    # FILTER 2: Drop rows with missing critical data
    df = df.dropna(subset=['mmsi', 'longitude', 'latitude', 'date_time'])
    print(f"  After dropping missing values: {len(df)}")
    
    # ADD FEATURES
    # df = _add_time_features(df, 'date_time')  # COMMENT OUT
    df = _course_features(df, 'cog')
    return df.sort_values('date_time').reset_index(drop=True)


def clean_radar(path):
    """
    Cleans Radar CSV file. Expected columns:
    targetId, status, longitude, latitude, sog, cog, signalLevel, dateTime, data, source_message
    """
    print(f"Loading Radar from: {path}")
    df = pd.read_csv(path)
    print(f"  Raw rows: {len(df)}")
    
    # Parse messy date format
    df['dateTime'] = df['dateTime'].apply(_parse_excel_datetime)
    
    # Keep only needed columns
    keep = ['targetId', 'longitude', 'latitude', 'sog', 'cog', 'signalLevel', 'dateTime']
    df = df[keep].copy()
    
    # Convert to numeric
    for c in ['longitude', 'latitude', 'sog', 'cog', 'signalLevel']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    # Same geographic filter as AIS
    df = df[(df['longitude'].between(128.5, 130.0)) & 
            (df['latitude'].between(34.5, 36.0))]
    print(f"  After geographic filter: {len(df)}")
    
    # Drop missing critical data
    df = df.dropna(subset=['targetId', 'longitude', 'latitude', 'dateTime'])
    print(f"  After dropping missing values: {len(df)}")
    
    # df = _add_time_features(df, 'dateTime')  # COMMENT OUT
    df = _course_features(df, 'cog')
    return df.sort_values('dateTime').reset_index(drop=True)


def haversine_m(lon1, lat1, lon2, lat2):
    """
    Calculates distance in meters between two GPS coordinates.
    Uses the haversine formula (accounts for Earth's curvature).
    """
    R = 6371000.0  # Earth radius in meters
    
    # Convert to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.sin(dlat/2)**2 + 
         np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2)
    
    return 2 * R * np.arcsin(np.sqrt(a))