from src.file_manager import FileManager
from loguru import logger

def landing_driver() -> None:  
    fm = FileManager()
    fm.emptyPersistentLandingZone()

    # # 2. Create Data Sources
    dataSources = ['OpenData', 'PortalDades']
    for ds in dataSources:
        fm.newDataSource(ds)

    # 3. Load all files from temporal landing zone to persistent zone
    ds1df1 = fm.persistFile("Opendata_2020.csv", "OpenData")
    ds1df2 = fm.persistFile("Opendata_2021.csv", "OpenData")
    ds2df1 = fm.persistFile("Portaldades_2020.csv", "PortalDades")
    ds2df2 = fm.persistFile("Portaldades_2021.csv", "PortalDades")