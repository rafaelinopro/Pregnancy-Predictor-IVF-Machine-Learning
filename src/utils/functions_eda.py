from utils.libreries import *

def df_attack(df, binary_target, binario = False):
    ''' Esta función nos da un resumen de las características del dataframe.
    Intenta tener un formato mas visual que el .info()'''
    import pandas as pd
    import numpy as np
    
    cols = df.columns

    # Defino variables para un print de la cantidad de filas y de columnas
    CANTIDAD_FILAS = df.shape[0]
    CANTIDAD_COLUMNAS = df.shape[1]

    # Defino listas de las columnas segun el tipo de datos. Tambien defino la longitud de dichas listas
    object_cols = list(df.select_dtypes('object').columns)
    integer_cols = list(df.select_dtypes('int').columns)
    float_cols = list(df.select_dtypes('float').columns)
    datetime_cols = list(df.select_dtypes('datetime').columns)
    len_object_cols = len(object_cols)
    len_integer_cols = len(integer_cols)
    len_float_cols = len(float_cols)
    len_datetime_cols = len(datetime_cols)

    # Creamos un diccionario de los nombres de columnas y su descripción. 
    col_dict = {}
    # for i in cols:
    #     descripcion_cols = input(f'Ingrese descripción de columna {i}:')
    #     col_dict[i] = descripcion_cols

    # Calculamos el porcentaje de missings values de cada columna, y ponemos en unalista aquellas que tienen mas del 30% de missings
    missings_cols = []
    
    for i in cols:
        porcentaje_missings = df[i].isnull().sum()/CANTIDAD_FILAS
        if porcentaje_missings > 0.3:
            missings_cols.append(i)

    # Calculamos la cantidad de valores unicos en cada columna, y el porcentaje del que mayor se repite sobre el total de filas.
    
    valor_unico_max = []
    porcentaje_unico_max = []

    for i in cols: 
        valor_unico_max.append(df[i].value_counts().index[0])
        porcentaje_unico_max.append(df[i].value_counts().max()/CANTIDAD_FILAS)
    
    data = [valor_unico_max,porcentaje_unico_max]
    df_porcentaje_unicos = pd.DataFrame(data, columns = cols)
    df_porcentaje_unicos = df_porcentaje_unicos.T

    # Hacemos una matriz de correlacion entre todas las columnas y detectamos aquellas columnas que tienen colinealidad con otra, y las guardamos en un set.
    df_correlacion = df.corr()
    df_correlacion = df_correlacion.abs()
    df_correlacion = df_correlacion[df_correlacion > 0.8]
    corr_filas = df_correlacion.index
    corr_columnas = df_correlacion.columns
    colinealidad = {}

    for k in corr_filas:
        for v in corr_columnas:
           if not pd.isna(df_correlacion.loc[k, v]):
                value = df_correlacion.loc[k, v]
                colinealidad[value] = (v)

    columnas_colineales = set(colinealidad.values())

    # Preguntamos si el df tiene target binario. Si es asi, calculamos el balance.
    
    if binario:
        balance_target = (df[binary_target].value_counts()[1] * 100) / (df[binary_target].value_counts()[1] + df[binary_target].value_counts()[0])

    print(f'Cantidad de filas: {CANTIDAD_FILAS}')
    print()
    print(f'Cantidad de columnas: {CANTIDAD_COLUMNAS}')
    print()
    print(f'Cantidad de type OBJECT: {len_object_cols}')
    print()
    print(f'Cantidad de type INTEGER: {len_integer_cols}')
    print()
    print(f'Cantidad de type FLOAT: {len_float_cols}')
    print()
    print(f'Cantidad de type DATETIME: {len_datetime_cols}')
    print()
    print(f'Columnas con porcentaje de missings mayor del 30%: {len(missings_cols)}')
    print(missings_cols)
    print()
    print(f'Columnas con colinealidad: {len(columnas_colineales)}')
    print(columnas_colineales)
    print()
    if binario:
        print(f'Balance target: {balance_target}')
        print()

    print('Porcentaje de valores únicos')
    pd.set_option('display.max_columns', None)
    return df_porcentaje_unicos.sort_values(by = 1, ascending = False).head(10)

    # return df, col_dict, missings_cols, len_object_cols, len_integer_cols, len_float_cols, len_datetime_cols, df_porcentaje_unicos, balance_target



# Función para renombrar columnas, poniendole numero de orden y dtype antes del nombre. Tambien lo pasamos a minusculas y reemplazamos los espacios por -.

def rename_columns(df):
    """
    Renames the columns of a given pandas dataframe with a new name that contains 
    the index of the column, its dtype, and its name in lower case with spaces 
    replaced by underscores. 

    :param df: A pandas dataframe.
    :return: None
    """
    new_columns = []
    for i, column in enumerate(df.columns):
        dtype = str(df[column].dtype)
        new_name = f"{i}_{dtype}_{column.lower().replace(' ', '_')}"
        new_columns.append(new_name)
    df.columns = new_columns


# Función para automatizar el get_dummies en varias columnas.

def dummies_object(df):
    """
    Generates dummy variables for all columns in a pandas dataframe that have a dtype of 'object'.
    
    Parameters:
    -----------
    df: pandas.DataFrame
        The input dataframe.
    
    Returns:
    --------
    pandas.DataFrame
        A new dataframe with dummy variables for all categorical columns.
    """
    col_list = [col for col in df.columns if df[col].dtype == 'object']
    for col in col_list:
        df = df.join(pd.get_dummies(df[col], prefix=col))
        df = df.drop(columns=[col])
    return df

def data_report(df):
    # Sacamos los NOMBRES
    cols = pd.DataFrame(df.columns.values, columns=["COL_N"])

    # Sacamos los TIPOS
    types = pd.DataFrame(df.dtypes.values, columns=["DATA_TYPE"])

    # Sacamos los MISSINGS
    percent_missing = round(df.isnull().sum() * 100 / len(df), 2)
    percent_missing_df = pd.DataFrame(percent_missing.values, columns=["MISSINGS (%)"])

    # Sacamos los VALORES UNICOS
    unicos = pd.DataFrame(df.nunique().values, columns=["UNIQUE_VALUES"])
    
    percent_cardin = round(unicos['UNIQUE_VALUES']*100/len(df), 2)
    percent_cardin_df = pd.DataFrame(percent_cardin.values, columns=["CARDIN (%)"])

    concatenado = pd.concat([cols, types, percent_missing_df, unicos, percent_cardin_df], axis=1, sort=False)
    concatenado.set_index('COL_N', drop=True, inplace=True)


    return concatenado.T

def variable_analysis(columna):
    import matplotlib.pyplot as plt
    if columna.dtype == 'object':
        plt.figure(figsize=(5, 3))
        value_counts = columna.value_counts()
        ax = value_counts.plot(kind='bar', color=['skyblue', 'lightgreen', 'coral', 'gold'], edgecolor='black')
        if any([isinstance(value, str) and len(value) > 4 for value in columna]):  
            plt.xticks(rotation=90)
        else:
            plt.xticks(rotation=0)
        
        plt.title(columna.name)
        for i, v in enumerate(value_counts):
            ax.text(i, v, str(v), ha='center', va = 'bottom', fontsize=8)
        plt.show()

        print(columna.name)
        print(f'La variable es de tipo: {columna.dtype}')
        print(f'Los valores únicos son: {columna.unique()}')
        print(f'La cantidad de missings es: {columna.isnull().sum()}')
    
    elif columna.dtype == 'int64':
        plt.figure(figsize=(5, 3))
        value_counts = columna.value_counts()
        ax = value_counts.plot(kind='bar', color=['skyblue', 'lightgreen', 'coral', 'gold'], edgecolor='black')
        plt.xticks(rotation=0)
        plt.title(columna.name)
        for i, v in enumerate(value_counts):
            ax.text(i, v, str(v), ha='center', va = 'bottom', fontsize=8)
        plt.show()

        print(columna.name)
        print(f'La variable es de tipo: {columna.dtype}')
        print(f'Los valores únicos son: {columna.unique()}')
        print(f'La cantidad de missings es: {columna.isnull().sum()}')
    
    elif columna.dtype == 'float64':
        if columna.isna().sum() > 0:
            columna_int = columna.dropna().astype('int64')            
            plt.figure(figsize=(5, 3))
            value_counts = columna_int.value_counts()
            ax = value_counts.plot(kind='bar', color=['skyblue', 'lightgreen', 'coral', 'gold'], edgecolor='black')
            plt.xticks(rotation=0)
            plt.title(columna.name)
            for i, v in enumerate(value_counts):
                ax.text(i, v, str(v), ha='center', va = 'bottom', fontsize=8)
            plt.show()

            print(columna.name)
            print(f'La variable es de tipo: {columna.dtype}')
            print(f'La cantidad de missings es: {columna.isnull().sum()}')
            print('La frecuencia de valores únicos es: ')
            print(columna.value_counts())
           

        else:
            columna.astype('int64')
            plt.figure(figsize=(5, 3))
            value_counts = columna.value_counts()
            ax = value_counts.plot(kind='bar', color=['skyblue', 'lightgreen', 'coral', 'gold'], edgecolor='black')
            plt.xticks(rotation=0)
            plt.title(columna.name)
            for i, v in enumerate(value_counts):
                ax.text(i, v, str(v), ha='center', va = 'bottom', fontsize=8)
            plt.show()

            print(columna.name)
            print(f'La variable es de tipo: {columna.dtype}')
            print(f'La cantidad de missings es: {columna.isnull().sum()}')
            print('La frecuencia de valores únicos es: ')
            print(columna.value_counts())


        # for patch, count in zip(patches, n):
        #     height = patch.get_height()
        #     plt.text(patch.get_x(), height, count, ha='center', va='bottom', fontsize=8)
    
    else:
        print(columna.name)
        print(f'La variable es de tipo: {columna.dtype}')
        print(f'La cantidad de missings es: {columna.isnull().sum()}')
        print('La frecuencia de valores únicos es: ')
        print(columna.value_counts())
       

def short_col_names(data):
    short_cols = []
    for i, col in enumerate(data.columns):
        variable_corta = f'c{i+1}'
        exec(f"{variable_corta} = data.columns[{i}]")
        short_cols.append(variable_corta)
    return short_cols