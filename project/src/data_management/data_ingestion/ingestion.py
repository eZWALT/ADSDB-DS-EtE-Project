import pandas as pd 
import duckdb 

from loguru import logger

def ingestion_driver() -> None:  
    logger.success("Files ingested!")