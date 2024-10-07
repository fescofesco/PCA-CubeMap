# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:40:55 2024

@author: felix

This script fetches cube data from specified URLs, processes the data, 
and saves the cardnames ("name") and the date of the 
latest change ("latestTmsp")in CSV format. 
f"cardlists_{cubename}.csv"


Example Usage:
    To read URLs from the 'fetch_cubedata.txt' file and fetch cube data:
    
    if __name__ == "__main__":
        urls = read_urls()  # Reads the URLs from the text file
        fetch_cube_data(urls)  # Fetch data from all the URLs and save the data
"""

import requests
import pandas as pd
from pathlib import Path
import os
import re
from datetime import datetime

def read_urls(file_path=None):
    """
    Reads URLs from the specified text file, filters out comments and empty lines,
    and constructs a list of cube API URLs.

    Args:
        file_path (Path): The path to the file containing cube data URLs.

    Returns:
        list: A list of formatted URLs to fetch cube data from.
    """
    urls_list = []
    
    if file_path == None:
        file_path = Path("Input/Cubedata/fetch_cubedata.txt")
    
    try:
        with file_path.open('r') as file:
            for line in file:
                line = line.strip()
                if not line.startswith("#") and line:  # Skip lines starting with '#' and empty lines
                    urls_list.append("https://cubecobra.com/cube/api/cubeJSON/" + line.replace("https://cubecobra.com/cube/overview/", ""))
    except Exception as e:
        print(f"Error reading URLs file: {e}")
    print(urls_list)

    return urls_list


def fetch_cube_data(urls_list=None, export_path = None, verbose =0):
    """
    Fetches cube data from the specified list of URLs. If a cube name's data
    file already exists, it skips downloading it.

    Args:
        urls_list (list, optional): A list of URLs to fetch data from. 
                                     If None, it defaults to a sample URL.
        export_path(Path, optional): Where the files shall be saved to.
    """
    if urls_list is None:
        urls_list = ["https://cubecobra.com/cube/api/cubeJSON/chrilix"]


    for url in urls_list:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                import_data = response.json()
                cubename = import_data.get("name", "unknown_cube")              
    
                # Remove backslashes
                cubename = cubename.replace("\\", "").replace("/","")
                
                # Remove all special characters (keeping only letters, numbers, and spaces)
                cubename = re.sub(r'[^A-Za-z0-9() ]+', '', cubename)
    
                print(cubename)
                
                datetimeYYYYMMDD = datetime.now().strftime('%Y%m%d')
                df = pd.json_normalize(import_data)
                df.to_json(export_path / f"json_{cubename}_{datetimeYYYYMMDD}.json", orient='records', lines=True)



                if verbose > 0:
                    print(f"Imported Cube Data from: {url}")

                try:
                    cube_df, latestTmsp = jsondata_to_df(import_data)
                    save_imported_data(cube_df, cubename, latestTmsp)
                except ValueError as e:
                    print(f"Error normalizing data from {url}: {e}")
            else:
                print(f"Failed to fetch data from {url}, Status Code: {response.status_code}")
        except Exception as e:
            print(f"Error fetching data from {url}: {e}")

def save_imported_data(data, default_filename, latestTmsp):
    """
    Saves the cleaned cube data to a CSV file.

    Args:
        data (DataFrame): The DataFrame containing cleaned cube data.
        default_filename (str): The base filename for the saved CSV.
    """
    
    filename = f"cardlist_{latestTmsp}_{default_filename}.txt"
    export_txt_path = Path("../Input/Cubedata") / filename

    try:
        # export_data(data, "windows-1250", export_csv_path)
        export_data(data, "utf-8", export_txt_path)
        print(f"Data saved to {filename}")
    except Exception as e:
        print(f"Error saving data: {e}")

def convert_to_iso_format(timestamp):
    try:
        # Check if the value is already in ISO 8601 format
        dt = pd.to_datetime(timestamp)
        return dt  # If it is, return as is
        

    except (ValueError, TypeError):
        # If not, assume it's in milliseconds and convert it to ISO 8601
        return pd.to_datetime(timestamp, unit='ms', utc=True)

def jsondata_to_df(data):
    normalized_df = pd.json_normalize(data['cards']['mainboard'])

    columns_to_display = [
        'details.name', 'date'
    ]

    available_columns = [col for col in columns_to_display if col in normalized_df.columns]
    filtered_df = normalized_df[available_columns].copy()
    missing_columns = [col for col in columns_to_display if col not in normalized_df.columns]
    for col in missing_columns:
        filtered_df.insert(0, col, value="")

    
    try:
        # Assume the 'date' field in data is in Unix timestamp (milliseconds)
        latestTmsp = pd.to_datetime(data['date'], unit='ms', utc=True)
        # Format the parsed date to 'dd mm YYYY'
        latestTmsp = latestTmsp.strftime('%d %m %Y')
    except (ValueError, TypeError) as e:
        # Handle cases where the date is invalid or cannot be parsed
        print(f"Error parsing date: {e}")
        latestTmsp = "Invalid Date"

    print(latestTmsp)
    # print(LatestTmsp = filtered_df['date'])

    rename_columns = {
        'details.name': 'name'
    }
    filtered_df.rename(columns=rename_columns, inplace=True)

    columns_order = [
        'name'
    ]
    filtered_df = filtered_df.reindex(columns=columns_order)

    return filtered_df, latestTmsp

def export_data(data, encoding_value="windows-1250", expo_path=None):
    """
    Exports the DataFrame to a TXT file with the specified encoding.

    Args:
        data (DataFrame): The DataFrame to export.
        encoding_value (str): The encoding to use when writing the TXT file.
        expo_path (str or Path): The path where the TXT file will be saved.
    """
    # Check if expo_path is None
    if expo_path is None:
        # Get the current working directory
        current_location = os.getcwd()
        # Generate the current datetime string
        datetime_str = datetime.now().strftime("%Y%m%d%H%M%S")
        # Create the default export path
        expo_path = os.path.join(current_location, f"cubecobra_export_{datetime_str}.txt")
        
        print("Saved to: ", expo_path)

    try:
        # Export the DataFrame to TXT
        data.to_csv(expo_path, index=False, sep='\t', encoding=encoding_value)
        print("Export successful!")
    except Exception as e:
        print(f"Error exporting data: {e}")
    
    

if __name__ == "__main__":

    # Define the relative path to the input file
    input_file_path = Path("../Input/fetch_cubedata.txt")
    urls = read_urls(input_file_path)
    export_path = Path("..//Input/Cubedata")
    fetch_cube_data(urls,export_path)
    
