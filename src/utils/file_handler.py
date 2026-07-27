"""
Utility functions for reading and writing JSON files/

These helpers are shared across all modules (HR,IT,Finance,Travel).

"""

import json 
from pathlib import Path 
from typing import Any 

def load_json(file_path:Path)->Any:
    """
    Load JSON data from a file.
    
    Args:
        file_path: Path to the JSON file.
        
    Returns:
        Parsed JSON data.
    """

    if not file_path.exists():
        return [] 

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {file_path.name}.") from exc

def save_json(file_path:Path,data:Any)->None:
    """
    Save data to a JSON file.
    
    Args:
        file_path:Path to the JSON file.
        data:Data to save. 
    """

    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path,"w",encoding='utf-8') as file:
        json.dump(data,file,indent=4)

