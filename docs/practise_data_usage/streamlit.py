import pandas as pd
from plotly import graph_objects as go
import plotly.express as px
import streamlit as st


file = '/home/lex/Downloads/1robot_2024.xlsx'


df = pd.read_excel(file, sheet_name=3, skiprows=3, 
                  )
df = df.drop(columns=['Unnamed: 1', 'Unnamed: 8'])
df = df[df['Unnamed: 0'].str.contains('округ')]


fig1 = px.bar(
    df, 
    y= [
        'Отсутствие финансово-хозяйственной деятельности',
       'Отсутствие необходимости \nв использовании для текущей деятельности организации',
       'Недостаток собственных денежных средств',
       'Недостаток \nфинансовой \nподдержки со стороны государства',
       'Недостаток квалифицированных специалистов', 'Другие \nпричины'
    ],
    x='Unnamed: 0'
)
fig1.update_layout(xaxis={'categoryorder':'total descending'})


st.plotly_chart(fig1)