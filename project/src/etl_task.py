from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Any, Dict
from loguru import logger

# Base Task class
class Task(ABC):
    @abstractmethod
    def run(self) -> None:
        """Run the task."""
        pass

# ETL Task class
class ETLTask(Task, BaseModel):
    config: Dict[str, Any]

    @abstractmethod
    def extract(self) -> Any:
        """Extract data from the source."""
        pass

    @abstractmethod
    def transform(self) -> Any:
        """Transform the extracted data."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Load the transformed data into the destination."""
        pass

    def run(self) -> None:
        """Run the ETL process."""
        logger.info(f"Running ETL task from {self.source} to {self.destination}")

        # Step 1: Extract
        data = self.extract()
        logger.info("Data extracted: {}", data)

        # Step 2: Transform
        transformed_data = self.transform(data)
        logger.info("Data transformed: {}", transformed_data)

        # Step 3: Load
        self.load(transformed_data)
        logger.info("Data loaded to destination.")
