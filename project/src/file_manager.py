import shutil
from pathlib import Path
from loguru import logger

class FileManager:
    def __init__(self):
        self.project_path = Path.cwd()
        self.temporallandingpath = self.project_path / "src" / "data_management" / "landing_zone" / "temporal"
        self.persistentlandingpath = self.project_path / "src" / "data_management" / "landing_zone" / "persistent"

    def newDataSource(self, name):
        try:
            new_folder_path = self.persistentlandingpath / name
            if not new_folder_path.exists():
                new_folder_path.mkdir()
                logger.info(f"LANDING ZONE: Data Source Folder '{name}' created.")
                return True
            else:
                logger.warning("LANDING ZONE: Data Source Folder already exists.")
                return False
        except Exception as e:
            logger.error(f"LANDING ZONE: ERROR. Data Source Folder creation failed. Reason: {e}")
            return False

    def persistFile(self, fileName, dataSource):
        try:
            fileName_noExtension = Path(fileName).stem
            temp_file_path = self.temporallandingpath / fileName

            if temp_file_path.exists():
                year_if_dataset = fileName_noExtension.split("_")[1]
                newFileName = f"{dataSource}_{year_if_dataset}.csv"
                dataSource_folder = self.persistentlandingpath / dataSource

                if not dataSource_folder.exists():
                    logger.error(f"LANDING ZONE ERROR: The data source folder {dataSource_folder} does not exist.")
                    return False

                persistent_file_path = dataSource_folder / newFileName

                if persistent_file_path.exists():
                    logger.error(f"LANDING ZONE ERROR: The file {persistent_file_path} already exists.")
                    return False

                shutil.copy(temp_file_path, persistent_file_path)
                logger.info(f"LANDING ZONE: File '{fileName_noExtension}' duplicated to the Persistent landing zone.")
                return True
            else:
                logger.error("LANDING ZONE ERROR: non-existing file in the temporal landing zone.")
                return False

        except Exception as e:
            logger.error(f"LANDING ZONE ERROR. Reason: {e}")
            return False

    def emptyTemporalLandingZone(self):
        try:
            if self.temporallandingpath.exists():
                for item in self.temporallandingpath.iterdir():
                    if item.is_file() or item.is_symlink():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                logger.info("LANDING ZONE: Temporal landing zone emptied.")
                return True
        except Exception as e:
            logger.error(f"LANDING ZONE ERROR: Temporal landing zone emptying failed. Reason: {e}")
            return False

    def emptyPersistentLandingZone(self):
        try:
            if self.persistentlandingpath.exists():
                for item in self.persistentlandingpath.iterdir():
                    if item.name != "landing_zone.ipynb":
                        if item.is_file() or item.is_symlink():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)
                logger.info("LANDING ZONE: Persistent landing zone emptied.")
                return True
            else:
                logger.error(f"LANDING ZONE ERROR: The directory {self.persistentlandingpath} does not exist.")
                return False
        except Exception as e:
            logger.error(f"LANDING ZONE ERROR: Failed to empty persistent landing zone. Reason: {e}")
            return False
