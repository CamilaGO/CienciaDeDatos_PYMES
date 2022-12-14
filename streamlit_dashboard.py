# =========================Librarias generales=========================
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import io
from streamlit_option_menu import option_menu
# =========================EDA libraries=========================
import lux
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
# ===================Preprocesado y modelado=========================
from mlxtend.frequent_patterns import association_rules, apriori # se importa la funciones necesarias para aplicar el algoritmo Apriori a nuestros datos
from scipy.stats import pearsonr
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import statsmodels.formula.api as smf


# =========================Funciones generales=========================
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def tabla(x):
    if x<=0:
        return 0
    if x>=1:
        return 1

df = pd.DataFrame()

# =========================Dashboard=========================
# ------------------ Menu ------------------
selected = option_menu(menu_title = None, 
    options = ["Limpieza", 'Exploración', 'Modelos'], 
    icons=['brush', 'bar-chart-line', 'briefcase-fill'], 
    menu_icon="cast", 
    default_index=0,
    orientation="horizontal")

if selected == "Limpieza":
  # ------------------ Bienvenida y Load Data Set ---------------------------
  st.title("Limpieza de Datos")
  uploaded_file = st.file_uploader("Sube tu archivo .csv")

  if uploaded_file is not None and ".csv" in uploaded_file.name:
      # evaluar si es .csv y cargar los datos a un data frame
      df = pd.read_csv(uploaded_file, encoding = 'latin-1') # load it with csv
      
      if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', inplace=True, axis=1)

      st.success('Datos cargados! 🎉')
      # ------------------  Explorar dataframe ------------------
      # ___________ exploracion general ___________
      st.text("")
      st.info('Exploración general')
      df.to_pandas()
      st.write(df.head(), df.shape) # primeras 5 filas del dataframe y sus dimensiones
      # informacion del dataframe
      buffer = io.StringIO()
      df.info(buf=buffer)
      s = buffer.getvalue()
      st.text(s)
      # estadisticas basicas
      st.write("\nEstadisticas básicas de las columnas")
      st.write(df.describe(include=object))
      #filas duplicadas
      st.write("\nTotal de filas duplicadas: ", df.duplicated().sum())
      # filas con valores nulos
      st.write("Total de filas con valores nulos: ", np.count_nonzero(df.isnull()))
      # ___________ exploracion enfocada en una variable ___________
      # seleccionar la variable a enfocarse en la exploracion
      st.text("")
      st.info('Exploración específica')
      var_explore = st.selectbox(
          "Selecciona una variable para enfocar la exploración: ",
          list(df.columns)
      )
      # contar valores unicos
      st.write('Valores únicos en la columna seleccionada: ')
      st.table(df[var_explore].value_counts().head())

      # ------------------  Limpiar dataframe ------------------
      st.text("")
      st.info('¡Limpieza de datos!')
      st.write('Selecciona las técnicas de limpieza que deseas realizar:')
      limp_1 = st.checkbox('Eliminar caracteres especiales')
      limp_2 = st.checkbox('Cambiar mayúsculas a minúsculas')
      limp_3 = st.checkbox('Convertir en mayúscula la primera letra de cada palabra')
      limp_4 = st.checkbox('Eliminar filas duplicadas')
      limp_5 = st.checkbox('Eliminar valores nulos')
      limp_6 = st.checkbox('Cambiar los tipos de datos')
      suma_limp = limp_1 + limp_2 + limp_3 + limp_4 + limp_5 + limp_6
      if (limp_1):
        # se eliminan los caracteres especiales
        # Como se ha observado todos los nombres tienen una secuencia de ';;;;;;;' al final que posiblemente fue un error al momento de 
        # guardar los datos. Procederemos a eliminar dichos caracteres y letras con tildes, el simbolo de dolar y la ñ
        df.replace(regex={';': '', 'ñ': 'n', '$.': '', 'Ñ': 'N', '#':'', '&':'', '%':'', '~':'', '`':'',

                          'á': 'a','é': 'e','í': 'i','ó': 'o','ú': 'u',
                          'ä': 'a','ë': 'e','ï': 'i','ö': 'o','ü': 'u',
                          'ã': 'a','ẽ': 'e','õ': 'o',
                          
                          'Á': 'A','É': 'E','Í': 'I','Ó': 'O','Ú': 'U',
                          'Ä': 'A','Ë': 'E','Ï': 'I','Ö': 'O','Ü': 'U',
                          'Ã': 'A','Ẽ': 'E', 'Õ':'O'} ,inplace=True)
                          
        # tambien se eliminan dichos caracteres en los nombres de las columnas        
        df.columns = df.columns.str.replace(";", "")
        df.columns = df.columns.str.replace("ñ", "n")
        df.columns = df.columns.str.replace("$.", "")
        df.columns = df.columns.str.replace("Ñ", "N")
        df.columns = df.columns.str.replace("#", "n")
        df.columns = df.columns.str.replace("%", "")
        df.columns = df.columns.str.replace('`','')
        df.columns = df.columns.str.replace('&','',)

        df.columns = df.columns.str.replace("á", "a")
        df.columns = df.columns.str.replace("é", "e")
        df.columns = df.columns.str.replace("í", "i")
        df.columns = df.columns.str.replace("ó", "o")
        df.columns = df.columns.str.replace("ú", "u")
        df.columns = df.columns.str.replace('ä', 'a')
        df.columns = df.columns.str.replace('ë', 'e')
        df.columns = df.columns.str.replace('ï', 'i')
        df.columns = df.columns.str.replace('ö', 'o')
        df.columns = df.columns.str.replace('ü', 'u')
        df.columns = df.columns.str.replace('ã', 'a')
        df.columns = df.columns.str.replace('ẽ', 'e')
        df.columns = df.columns.str.replace('õ', 'o')

        df.columns = df.columns.str.replace("Á", "a")
        df.columns = df.columns.str.replace("É", "e")
        df.columns = df.columns.str.replace("Í", "i")
        df.columns = df.columns.str.replace("Ó", "o")
        df.columns = df.columns.str.replace("Ú", "u") 
        df.columns = df.columns.str.replace('Ä', 'A')
        df.columns = df.columns.str.replace('É', 'E')
        df.columns = df.columns.str.replace('Í', 'I')
        df.columns = df.columns.str.replace('Ó', 'O')
        df.columns = df.columns.str.replace('Ú', 'U')
        df.columns = df.columns.str.replace('Ã', 'A')
        df.columns = df.columns.str.replace('Ẽ', 'E')
        df.columns = df.columns.str.replace('Õ','O')
    
      if (limp_2):
        # se pasa todo a minuscula
        for x in df.columns:
          if df[x].dtype == 'object':
            df[x] = df[x].str.lower()

      if (limp_3):
        # convertir en mayuscula la primera letra de cada palabra
        #La primera letra de cada palabra se pone en mayuscula para mayor estetica
        for x in df.columns:
          if df[x].dtype == 'object':
            df[x] = df[x].str.title()

      if (limp_4):
        #eliminar filas duplicadas
        df = df.drop_duplicates()

      if (limp_5):
        #eliminar filas con valores nulos (NaN)
        df = df.dropna()
      
      if (limp_6):
        #cambiar tipos de valores
        # se pide el nombre de la variable que contiene el monto vendido y la fecha
        monto_var = st.multiselect(
            "Selecciona la variable del monto vendido",
            list(df.columns) 
        )

        # Selection variable de productos
        fecha_var = st.multiselect(
            "Selecciona la variable de la fecha",
            list(df.columns)
        )

        if len(monto_var)==1 and len(fecha_var)==1:
          # Para el monto se convierte la variable a tipo numérico (int64) ya que es tipo texto (object)
          df[monto_var[0]].replace(regex={'Q.': ''} ,inplace=True)
          df[monto_var[0]] = pd.to_numeric(df[monto_var[0]])

          # Para la fecha se convierte la variable a tipo date (ns) ya que es tipo texto (object)
          df[fecha_var[0]] = pd.to_datetime(df[fecha_var[0]])
        else:
          st.warning("Selecciona una variable en cada campo")

      if (suma_limp>0):
        if st.button('Ver resultados de limpieza'):
          st.success('Limpieza terminada! 🎉')
          df.to_pandas()
          st.markdown('Conjunto de datos limpios:')
          st.write(df.head())
          # informacion del dataframe
          buffer = io.StringIO()
          df.info(buf=buffer)
          s = buffer.getvalue()
          st.text(s)

          csv_clean = convert_df(df)
          st.download_button(
              label="Descargar datos limpios",
              data=csv_clean,
              file_name='datosLimpios.csv',
              mime='text/csv',
          )
      else:
        st.warning('Debes seleccionar mínimo 1 técnica de limpieza')
  else:
   # si no sube archivo de datos para limpiar
    st.warning("Por favor, sube tu archivo .csv antes de continuar")

elif selected == "Exploración":
  # ------------------ Bienvenida y Load Data Set ---------------------------
  st.title("Exploración de Datos")
  uploaded_clean_file = st.file_uploader("Sube tu archivo datosLimpios.csv")

  if uploaded_clean_file is not None:
    # Cargar los datos a un data frame
    #df_clean = pd.read_csv(uploaded_clean_file, encoding = 'latin-1') # load it with csv
    df = pd.read_csv(uploaded_clean_file, encoding = 'utf-8')
    df.drop('Unnamed: 0', inplace=True, axis=1)

    # ------------------ Creacion y descarga de HTMLs ---------------------------
    # el usuario elije que reporte EDA generar
    reporte_type = st.radio(
      "Reporte:",
      ('Lux', 'PandasProfiling'))

    if reporte_type == 'Lux':
      # ******* Lux **********
      st.write('Reporte interactivo Lux')
      html_content = df.save_as_html('reporteLux.html',output=True)
      st.download_button(
        label='Descargar reporte general HTML',
        data=html_content,
        file_name='reporteLux.html',
        mime='text/html'
      ) 
      st.write("Enfócate en una, dos o tres variables")
      # Selection of Columns
      columns_options = st.multiselect(
          "Selecciona las variables",
          list(df.columns),
      )
      df.intent = columns_options
      html_content = df.save_as_html('reporteLux_columselec.html',output=True)
      st.download_button(
        label='Descargar reporte enfocado HTML',
        data=html_content,
        file_name='reporteLux_columselec.html',
        mime='text/html'
      ) 
      df.to_pandas()
      
    else:
      # *****************  Pandas Profiling *********************
      df.to_pandas()
      st.write('Reporte estático PandasProfiling')
      #profile = ProfileReport(df, title='Pandas Report')
      #lux.config.default_display = "pandas"
      #df.to_pandas()
      pr = ProfileReport(df)

      #st_profile_report(pr)

      export=pr.to_html()
      st.download_button(label="Descargar reporte general HTML",
      data=export, 
      file_name='reportePandasProfiling.html',
      mime='text/html')
  else:
    # si no sube archivo de los datos limpios
    st.warning("Por favor, sube tu archivo antes de continuar")

elif selected == "Modelos":
  
  # ------------------ Bienvenida y Load Data Set ---------------------------
  st.title("Modelos básicos")
  uploaded_clean_file = st.file_uploader("Sube tu archivo datosLimpios.csv")

  if uploaded_clean_file is not None:
    # Cargar los datos a un data frame
    df = pd.read_csv(uploaded_clean_file, encoding = 'utf-8')
    df.drop('Unnamed: 0', inplace=True, axis=1)
    df.to_pandas()

    # ----------------- Solicitar variables para preparacion --------------------
    # Selection variable de fecha
    date_var = st.multiselect(
        "Selecciona la variable de fecha",
        list(df.columns) 
    )

    # Selection variable de productos
    prod_var = st.multiselect(
        "Selecciona la variable de producto",
        list(df.columns)
    )

    if len(prod_var)==1 and len(date_var)==1:
      # Extraer la información de la variable fecha
      df["Anio"]=pd.to_datetime(df[date_var[0]]).dt.year
      df["Mes"]=pd.to_datetime(df[date_var[0]]).dt.month
      df["Fecha_Dia"]=pd.to_datetime(df[date_var[0]]).dt.day
      df["Dia"]=pd.to_datetime(df[date_var[0]]).dt.weekday
      
      #### Categorizar años y agregar esta nueva columna a la base de datos
      df['anio_categoria'] = df['Anio'].astype('category')
      anios = df['Anio'].unique() # obtener los productos unicos
      # Establecer la categorizacion de los años
      df['anio_categoria'] = df['anio_categoria'].cat.set_categories(anios, ordered = True)
      categorias_anios = dict(enumerate(df['anio_categoria']))
      # Modificar la columna de la categoria con el numero del año segun las categorias creadas
      df['anio_categoria'] = df.anio_categoria.cat.codes

      #### Apartir de la fecha categorizar meses y agregar esta nueva columna a la base de datos
      df['Mes'].astype('category')

      #### Apartir de la fecha categorizar días y agregar esta nueva columna a la base de datos
      df['Dia'].astype('category')

      #### Categorizar productos y agregar esta nueva columna a la base de datos
      df['prod_categoria'] = df[prod_var[0]].astype('category')
      productos = df[prod_var[0]].unique() # obtener los productos unicos
      # Establecer la categorizacion de los productos
      df['prod_categoria'] = df['prod_categoria'].cat.set_categories(productos, ordered = True)
      categorias_productos = dict(enumerate(df['prod_categoria']))
      # Modificar la columna de la categoria con el numero del producto segun las categorias creadas
      df['prod_categoria'] = df.prod_categoria.cat.codes

      st.success('Datos categorizados! 🎉')
      df.to_pandas()
      st.write(df)
    else:
      st.warning('Elige 1 variable en cada campo')

    # ------------------ Menu de modelos ---------------------------
    modelo_type = st.radio(
      "Elige un modelo:",
      ('Asociación', 'Predicción'))
    
    # ------------------ Creacion y despliegue de modelos ---------------------------
    if modelo_type == 'Asociación':
      # ******* Reglas de Asociación **********
      st.text("")
      st.info('Modelo de Asociación')
      # ___________ Entrenamiento ____________________________
      # Selection variable de de factura o transaccion
      factura_var = st.multiselect(
          "Selecciona la variable identificadora de factura o transaccion",
          list(df.columns) 
      )
      if len(factura_var)==1:
        datos_apriori=df.groupby([factura_var[0],prod_var[0]])[prod_var[0]].count().reset_index(name="Cantidad")
        pivote_apriori = datos_apriori.pivot_table(index=factura_var[0],columns=prod_var[0],values="Cantidad",aggfunc="sum").fillna(0) # se le da una estructura de tabla pivote a la tabla anterior
        # en cada transacción se especifica cuatos productos el cliente adquirió para poder comenzar a ver las combinaciones de productos más recurrentes

        ### Se crea esta funcion para convertir todos los valores de la tabla pivote a 1 y 0, ya que son los valores aceptados por el algoritmo apriori
        tabla_pivote = pivote_apriori.applymap(tabla) # recorre cada casilla de la tabla pivote ejecutando la funcion "tabla"
        apriori_info = apriori(tabla_pivote,min_support=0.01,use_colnames=True)  # Obtenemos la confianza de cada combinacion de productos encontrada

        # ___________ Resultado ________________________________
        # se establecen las reglas de asociacion a los datos obtenidos anteriormente para conocer la combinación de productos más frecuente en las transacciones de los clientes
        reglas = association_rules(apriori_info, metric = "lift", min_threshold = 1) 
        final_assoc = reglas.sort_values("confidence",ascending=False)
        #st.write(reglas)
        st.subheader('Top 10 de combinaciones entre productos')
        st.write(final_assoc.head(10))

      else:
        st.warning("Elige una variable identificadora de factura o transaccion")

      
    elif modelo_type == 'Predicción':
      # ******* Predicción de demanda **********
      st.text("")
      st.info('Modelo predicción de demanda')
      pred_type = st.radio(
      "Predicciones de demanda por:",
      ('Año', 'Mes', 'Dia'))
      if pred_type == 'Año':
        pred_type = 'Anio'

      # ___________ Entrenamiento ____________________________
      # graficar las ventas por año
      ven_an = df.groupby([pred_type]).size().reset_index(name='Ventas')
      
      st.subheader("Cantidad de ventas por año")
      st.write(ven_an)

      # Asignamos nuestra variable de entrada X para entrenamiento y las etiquetas Y.
      dataX =ven_an[[pred_type]]
      X_train = np.array(dataX)
      y_train = ven_an['Ventas'].values

      regr = linear_model.LinearRegression() # Creamos el objeto de Regresión Linear
      
      regr.fit(X_train, y_train) # Entrenamos nuestro modelo

      y_pred = regr.predict(X_train) # Hacemos las predicciones que en definitiva una línea (en este caso, al ser 2D)

      # ___________ Resultado ________________________________
      st.text("")
      st.subheader("Predicción de demanda")
      fecha_pred = st.text_input('Introduce el año/mes/dia del que quieres predecir la demanda: ')
      if fecha_pred:
        y_Dosmil = regr.predict([[int(fecha_pred)]])
        st.write("\nPrediccion de demanda: ", (int(y_Dosmil)))
      else:
        st.warning("Introduce un año válido para predecir")


    

