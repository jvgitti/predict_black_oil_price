import pandas as pd
import plotly.express as px
import requests as r
from datetime import datetime

from PIL import Image
import streamlit as st

API_URL = 'https://api-predict-black-oil-price.onrender.com'

st.title('Análise para predição dos valores do barril de petróleo')

tab_0, tab_1 = st.tabs(['Análise Exploratória', 'Modelo'])

last_trained_date_response = r.get(API_URL + '/last_trained_date')
last_trained_date = last_trained_date_response.json()
last_trained_date = datetime.strptime(last_trained_date, '%Y-%m-%d').date()


with tab_0:
    """
    Antes de predizer o valor, devemos entender o contexto inserido e qual o comportamento da série em questão. No gráfico 
    abaixo, podemos visualizar - em um primeiro momento - o comportamento dos dados ao longo dos anos.
    """

    image = Image.open('images/historico.png')
    st.image(image)
    
    """
    Agora que entedemos a tendência geral, é importante analisar a decomposição sazonal da série.
    Para isso decompomos a série temporal para uma sazonalidade de 1 ano:
    """

    image = Image.open('images/seasonal_decompose.png')
    st.image(image)

    """
    Aqui podemos observar com clareza a tendência geral (gráfico 2), a sazonalidade (gráfico 3) e os resíduos (gráfico 4).

    Ao trabalhar com séries temporais - dependendo do modelo selecionado - é importate entender se a série é estacionaria ou nao-estacionaria. O teste Augmented Dickey-Fuller
    nos ajuda a entender se o conjunto em questao é ou nao estacionarios:
    """
    f"""
    Aplicando-se o teste de Dickey-Fuller, temos um valor de P-value = 0.25. Dessa maneira, não podemos rejeitar a hipótese nula, o que significa
    que temos uma série não estacionária.
    """
    """
    Abaixo temos uma representação visual da média movel em relaçao ao valores:
    """

    image = Image.open('images/media_movel.png')
    st.image(image)

    """
    Considerando que estamos trabalhando com uma série atualmente não-estacionaria, a primeira coisa que precisamos fazer é transforma-la. 
    Para isso aplicamos a primeira derivada:
    """

    image = Image.open('images/primeira_derivada.png')
    st.image(image)

    """
    Nesse novo formato, aplicamos novamente o teste de Dickey-Fuller, e chegamos a um valor de P-value muito próximo a 
    0, de modo que podemos rejeitar a hipótese nula, e considerar que a série agora é estacionária.
    """

    """
    Após todas as análises, testamos diversos modelo de série temporal. O que obteve melhor resultado, foi o modelo MSTL,
    juntamente com o Naive, utilizando uma sazonalidade anual, mensal e semanal.
    Abaixo segue o teste considerando um período de previsão de aproximadamente 1 ano. 
    """

    """
    WMAPE observado = 15.47%
    """

    image = Image.open('images/teste_modelo.png')
    st.image(image)

    """
    Link para a fontes das análises: https://github.com/jvgitti/predict_black_oil_price/blob/main/analytics.ipynb
    """

with tab_1:
    f"""
    Última data observada: {last_trained_date.strftime('%d/%m/%Y')}
    """

    if st.button('Atualizar dados e modelo'):
        response = r.post(API_URL + '/update_data_and_model')
        if response.status_code == 200:
            st.success(response.json()['status'])

    date_to_predict = st.date_input("Selecione a data para previsão:", value=last_trained_date)

    if last_trained_date > date_to_predict:
        st.warning('Escolha uma data maior que a última data observada', icon="⚠️")
    elif last_trained_date < date_to_predict:
        predict_response = r.post(API_URL + '/predict', json=dict(date=str(date_to_predict)))
        if predict_response.status_code != 200:
            st.error('Erro ao realizar predição', icon="!️")
        else:
            df_predict = pd.DataFrame(predict_response.json())

            last_day_predicted = df_predict['date'].max()
            value_last_day_predict = df_predict[df_predict.date == last_day_predicted]['value'].iloc[0]

            last_day_predicted = datetime.strptime(last_day_predicted, '%Y-%m-%d').strftime('%d/%m/%Y')

            f"""
                        Valor predito para o dia {last_day_predicted}: US$ {value_last_day_predict:.2f}
            """

            fig = px.line(df_predict, x='date', y='value', line_shape='linear')

            fig.update_layout(
                title='Preço do Petróleo Bruto - Predição do valor',
                xaxis_title='Data',
                yaxis_title='Valor (US$)',
                legend=dict(
                    x=0,
                    y=-0.2,
                    orientation="h",
                )
            )
            st.plotly_chart(fig, use_container_width=True)
