import os
import pandas as pd
from loguru import logger
from pathlib import Path
import numpy as np

#Definition of paths
project_path = Path.cwd()
rawdata_path = project_path / "src" / "data_management" / "data_ingestion" / "rawdata"
temporal_landing_path = project_path / "src" / "data_management" / "landing_zone" / "temporal"

preuspath = rawdata_path / "artificial_portaldades_lloguerpreus_2024.csv"
recomptepath = rawdata_path / "artificial_portaldades_lloguerrecompte_2024.csv"

portaldades2020_path = temporal_landing_path  / "Portaldades_2020.csv"
portaldades2021_path = temporal_landing_path / "Portaldades_2021.csv"

opendata2020_path = temporal_landing_path / "Opendata_2020.csv"
opendata2021_path = temporal_landing_path / "Opendata_2021.csv"



def ingestion_driver() -> None:  
    etl_ingestion_opendata()
    etl_ingestion_portaldades()

def etl_ingestion_opendata(): 
    dataframes = {}
    for filename in os.listdir(rawdata_path):
        if  filename.endswith('.csv'):
            name = filename.split('.')[0]
            df = pd.read_csv(rawdata_path / filename)
            globals()['df_' + name] = df    #change df name to filename without extension
            dataframes['df_' + name] = df

    if not dataframes:
        logger.warning("No data files were loaded from raw data path.")
        return

    logger.info('Dataframes loaded:', [name for name in dataframes.keys()])

    """## 2. Create "Seccio censal" column to perform joins"""

    seccionswithAEB = pd.DataFrame()
    df = dataframes['df_artificial_opendata_equivalencies']

    seccionswithAEB['Seccio_Censal'] = df['SECC_CENS']
    seccionswithAEB['AEB'] = df['AEB']

    for df in dataframes:
        if 'Seccio_Censal' not in dataframes[df].columns and 'opendata' in df:
            dataframes[df] = pd.merge(dataframes[df], seccionswithAEB, on='AEB', how='left')


    """## 3. Merge datasets"""

    # Merge datasets with equal columns
    df_2020 = pd.DataFrame()
    df_2021 = pd.DataFrame()

    for df in dataframes:
        if 'demografia' in df:
            continue
        if '2020' in df:
            if df_2020.shape[0] == 0:
                df_2020 = dataframes[df]
                continue
            samecols = [col for col in df_2020.columns if col in dataframes[df].columns]
            df_2020 = df_2020.merge(dataframes[df], on=['Any', 'Codi_Districte', 'Nom_Districte', 'Codi_Barri', 'Nom_Barri', 'Seccio_Censal'])

        if '2021' in df:
            if df_2021.shape[0] == 0:
                df_2021 = dataframes[df]
                continue
            samecols = [col for col in df_2021.columns if col in dataframes[df].columns]
            df_2021 = df_2021.merge(dataframes[df], on=['Any', 'Codi_Districte', 'Nom_Districte', 'Codi_Barri', 'Nom_Barri', 'Seccio_Censal'])

    logger.info("Merging datasets by year completed.")

    df_2020 = df_2020.rename(columns={'Import_Euros_x': 'Import_Renda_Disponible', 'Import_Euros_y':'Import_Renda_Neta', 'Import_Renda_Bruta_€':'Import_Renda_Bruta' })
    df_2021 = df_2021.rename(columns={'Import_Euros_x': 'Import_Renda_Disponible', 'Import_Euros_y':'Import_Renda_Neta', 'Import_Renda_Bruta_€':'Import_Renda_Bruta' })

    # Merge demografic data with the big dataset by using the mean value of age
    #Compute the mean value of the age of each seccio censal.
    def computeMeanAge(df):
        if 'Data_Referecia' in df.columns:
            df.drop(columns=['Data_Referencia', 'Nom_Districte', 'Nom_Barri' ], inplace = True)
        df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce')
        df = df.groupby(['Codi_Districte', 'Codi_Barri','AEB', 'Seccio_Censal']).apply(lambda x: pd.Series({
            'Edat_Mitjana': (x['Valor'] * x['EDAT_1']).sum() / x['Valor'].sum(),
        })).reset_index()
        df.drop(columns=['AEB'], inplace=True)
        return df

    demografic_df_2020 = computeMeanAge(dataframes['df_artificial_opendata_demografia_2020'])
    demografic_df_2021 = computeMeanAge(dataframes['df_artificial_opendata_demografia_2021'])
    logger.info("Mean age computed")


    # 2.2 Merge demografic datasets with the whole data
    df_2020 = df_2020.merge(demografic_df_2020, on=['Codi_Districte', 'Codi_Barri', 'Seccio_Censal'], how='left')
    df_2021 = df_2021.merge(demografic_df_2021, on=['Codi_Districte', 'Codi_Barri', 'Seccio_Censal'], how='left')

    """## 4. Export 2020 and 2021 datasets to temporal landing zone"""

    # Export merged dataframes
    if not df_2020.empty and not df_2021.empty:
        df_2020.to_csv(opendata2020_path, index=False)
        df_2021.to_csv(opendata2021_path, index=False)
    else:
        logger.warning("Merged dataframes for 2020 or 2021 are empty.")

    logger.success("Ingestion of opendata successfully completed!")


def etl_ingestion_portaldades():
    preus = pd.read_csv(preuspath)
    recompte = pd.read_csv(recomptepath)

    """## 2. Preprocess the Preus dataset"""

    # keep only the columns containing 2020 and 2021 in the name
    preus = preus.filter(regex='2020|2021|Territori|Tipus')

    #delete the first 3 rows
    preus = preus.iloc[3:]

    #all the rows with "Barri" in "Tipus de territori" have a new column called 'Districte' that contains the districte of the row.
    # Rows between two "Districte" rows are the barris of the first apperance districte.

    preus['Districte'] = preus['Territori']
    preus['Districte'] = preus['Districte'].where(preus['Tipus de territori'] == 'Districte')
    preus['Districte'] = preus['Districte'].ffill()

    #split the data in two dataframes, one with 2020 data and the other with 2021 data
    preus_2020 = preus.filter(regex='2020|Territori|Tipus|Districte')
    preus_2021 = preus.filter(regex='2021|Territori|Tipus|Districte')

    # delete the rows with 'Districte' in 'Tipus de territori'
    preus_2020 = preus_2020[preus_2020['Tipus de territori'] != 'Districte']
    preus_2021 = preus_2021[preus_2021['Tipus de territori'] != 'Districte']

    #delete the 'Tipus de territori' column
    preus_2020 = preus_2020.drop(columns=['Tipus de territori'])
    preus_2021 = preus_2021.drop(columns=['Tipus de territori'])

    # add year column with the year of the data
    preus_2020['Any'] = 2020
    preus_2021['Any'] = 2021

    #rename the columns of the dataframes
    preus_2020.columns = ['Barri', 'Preu_1Trim', 'Preu_2Trim', 'Preu_3Trim', 'Preu_4Trim', 'Districte', 'Any']
    preus_2021.columns = ['Barri', 'Preu_1Trim', 'Preu_2Trim', 'Preu_3Trim', 'Preu_4Trim', 'Districte', 'Any']


    preus_2020 = preus_2020[['Any', 'Districte', 'Barri', 'Preu_1Trim', 'Preu_2Trim', 'Preu_3Trim', 'Preu_4Trim']]
    preus_2021 = preus_2021[['Any', 'Districte', 'Barri', 'Preu_1Trim', 'Preu_2Trim', 'Preu_3Trim', 'Preu_4Trim']]

    # replace the '-' values with NaN
    preus_2020 = preus_2020.replace('-', np.nan)
    preus_2021 = preus_2021.replace('-', np.nan)

    # convert the columns with prices to float
    preus_2020['Preu_1Trim'] = preus_2020['Preu_1Trim'].astype(float)
    preus_2020['Preu_2Trim'] = preus_2020['Preu_2Trim'].astype(float)
    preus_2020['Preu_3Trim'] = preus_2020['Preu_3Trim'].astype(float)
    preus_2020['Preu_4Trim'] = preus_2020['Preu_4Trim'].astype(float)

    preus_2021['Preu_1Trim'] = preus_2021['Preu_1Trim'].astype(float)
    preus_2021['Preu_2Trim'] = preus_2021['Preu_2Trim'].astype(float)
    preus_2021['Preu_3Trim'] = preus_2021['Preu_3Trim'].astype(float)
    preus_2021['Preu_4Trim'] = preus_2021['Preu_4Trim'].astype(float)

    # Mean of the 4 quarters of 2020 and 2021 for row in a new column called 'Preu_mitja'
    preus_2020['Preu_mitja'] = preus_2020[['Preu_1Trim', 'Preu_2Trim', 'Preu_3Trim', 'Preu_4Trim']].mean(axis=1)
    preus_2021['Preu_mitja'] = preus_2021[['Preu_1Trim', 'Preu_2Trim', 'Preu_3Trim', 'Preu_4Trim']].mean(axis=1)

    logger.info("Preus dataset preprocessed")


    """## 3. Preprocess the "Recompte" dataset"""

    recompte = recompte.filter(regex='2020|2021|Territori|Tipus')
    #Delete the first 3 rows
    recompte = recompte.iloc[3:]

    #all the rows with "Barri" in "Tipus de territori" have a new column called 'Districte' that contains the districte of the row.
    # Rows between two "Districte" rows are the barris of the first apperance districte.

    recompte['Districte'] = recompte['Territori']
    recompte['Districte'] = recompte['Districte'].where(recompte['Tipus de territori'] == 'Districte')
    recompte['Districte'] = recompte['Districte'].ffill()

    #split the data in two dataframes, one with 2020 data and the other with 2021 data
    recompte_2020 = recompte.filter(regex='2020|Territori|Tipus|Districte')
    recompte_2021 = recompte.filter(regex='2021|Territori|Tipus|Districte')

    # delete the rows with 'Districte' in 'Tipus de territori'
    recompte_2020 = recompte_2020[recompte_2020['Tipus de territori'] != 'Districte']
    recompte_2021 = recompte_2021[recompte_2021['Tipus de territori'] != 'Districte']

    #delete the 'Tipus de territori' column
    recompte_2020 = recompte_2020.drop(columns=['Tipus de territori'])
    recompte_2021 = recompte_2021.drop(columns=['Tipus de territori'])

    # add year column with the year of the data
    recompte_2020['Any'] = 2020
    recompte_2021['Any'] = 2021

    #rename the columns of the dataframes
    recompte_2020.columns = ['Barri', 'Recompte', 'Districte', 'Any']
    recompte_2021.columns = ['Barri', 'Recompte', 'Districte', 'Any']

    recompte_2020 = recompte_2020[['Any', 'Districte', 'Barri', 'Recompte']]
    recompte_2021 = recompte_2021[['Any', 'Districte', 'Barri', 'Recompte']]

    recompte_2020 = recompte_2020.replace('-', np.nan)
    recompte_2021 = recompte_2021.replace('-', np.nan)

    recompte_2020['Recompte'] = recompte_2020['Recompte'].astype(float)
    recompte_2021['Recompte'] = recompte_2021['Recompte'].astype(float)

    logger.info("Recompte dataset preprocessed")

    """## 4. Merge and save the datasets to the temporal landing zone"""

    #merge the two dataframes
    merged_2020 = pd.merge(preus_2020, recompte_2020, on=['Any', 'Districte', 'Barri'])
    merged_2021 = pd.merge(preus_2021, recompte_2021, on=['Any', 'Districte', 'Barri'])
    logger.info(f"Merged datasets for 2020 with {len(merged_2020)} records and 2021 with {len(merged_2021)} records.")

    #save the dataframes to csv
    merged_2020.to_csv(portaldades2020_path, index=False)
    merged_2021.to_csv(portaldades2021_path, index=False)

    logger.success("Ingestion of Portaldades successfully completed!")