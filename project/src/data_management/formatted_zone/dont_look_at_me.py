# Example implementation of a specific ETL task
class MyETLTask(ETLTask):
    def extract(self) -> Any:
        # Implement the logic to extract data from the source
        return {"key": "value"}  # Example data

    def transform(self, data: Any) -> Any:
        # Implement the transformation logic
        return {k: v.upper() for k, v in data.items()}  # Example transformation

    def load(self, data: Any) -> None:
        # Implement the logic to load data to the destination
        logger.info("Loading data to {}: {}", self.destination, data)

