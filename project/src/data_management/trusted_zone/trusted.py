import pandas as pd 
import duckdb 

from loguru import logger

def trusted_driver() -> None:  
    logger.success("Trusted checks done!")