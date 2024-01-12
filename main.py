from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import requests as r
from datetime import datetime

import streamlit as st

API_URL = 'https://api-predict-black-oil-price.onrender.com'

st.title('Análise para predição dos valores do barril de petróleo')

tab_0, tab_1 = st.tabs(['Análise Exploratória', 'Modelo'])

df = pd.read_csv('oil_price.csv', sep='\t', names=['ds', 'y'])

df['ds'] = pd.to_datetime(df['ds'])
df['y'] = df['y'].str.replace(',', '.').astype(float)
df = df.set_index('ds')

ad_fuller = adfuller(df.y.values)
p_value = ad_fuller[1]

last_trained_date_response = r.get(API_URL + '/last_trained_date')
last_trained_date = last_trained_date_response.json()
last_trained_date = datetime.strptime(last_trained_date, '%Y-%m-%d').date()


with tab_0:
    """
    O IBOVESPA (Índice da Bolsa de Valores de São Paulo) é o principal índice de ações do mercado de capitais brasileiro e 
    serve como indicador do desempenho médio das cotações das ações mais negociadas e mais representativas do mercado brasileiro. 
    Ele é utilizado tanto para entender o comportamento do mercado acionário brasileiro como um todo, quanto como referência 
    para investimentos. Um índice forte pode indicar um mercado em alta, com crescimento econômico e confiança dos investidores, 
    enquanto um índice fraco pode sinalizar o contrário.

    Antes de predizer o fechamento da base, devemos entender o contexto inserido e qual o comportamento da série em questão. No gráfico 
    abaixo, podemos visualizar - em um primeiro momento - o comportamento dos dados ao longo dos anos.
    De maneira geral podemos identificar uma tendência de crescimento, porém em 2020 temos uma grande queda no fechamento da bolsa, marcado
    por um dos maiores eventos já ocorridos na história.
    """

    plt.figure()
    sns.lineplot(data=df, x='ds', y='y', color='green')
    plt.xlabel('Ano')
    plt.ylabel('Valor (US$)')
    plt.title('Preço - Petróleo Bruto')
    st.pyplot(plt)

    """
    Principais quedas na bolsa IBOVESPA
    """

    """
    A bolsa de valores foi marcada por uma série de fatores, e foi diretamente impactada por eles, abaixo temos uma representação de 5 quedas enfrentadas
    pelo IBOVESPA. De acordo com uma notícia divulgada pela pr[opria B3: "A maior queda, de 22,26%, foi registrada no dia 21 de março de 1990, quando o Plano Collor foi anunciado. 
    Recentemente, a maior queda foi de 13,92%, em 16 de março de 2020, repercutindo a incerteza diante da pandemia."
    """
    
    st.image("https://www.b3.com.br/data/files/42/20/55/D4/E0AB8810C7AB8988AC094EA8/Linha%20do%20Tempo%20Ibovespa%20B3.png")
    """
    Fonte: https://www.b3.com.br/pt_br/noticias/ibovespa-b3-completa-55-anos-veja-10-curiosidades-sobre-o-indice-mais-importante-do-mercado-de-acoes-brasileiro.htm
    """
    
    """
    Agora que entendemos alguns fatores responsáveis pelas maiores quedas da bolsa e também a tendência geral que temos, é importante analisar a decomposição sazonal da série.
    Para isso decompomos a série temporal para uma sazonalidade de 1 ano:
    """

    plt.figure()
    resultados = seasonal_decompose(df, period=247)
    fig, axes = plt.subplots(4, 1, figsize=(15, 10))
    resultados.observed.plot(ax=axes[0])
    axes[0].set_xlabel('Ano')
    resultados.trend.plot(ax=axes[1])
    axes[1].set_xlabel('Ano')
    resultados.seasonal.plot(ax=axes[2])
    axes[2].set_xlabel('Ano')
    resultados.resid.plot(ax=axes[3])
    axes[3].set_xlabel('Ano')
    plt.tight_layout()
    st.pyplot(plt)

    """
    Aqui podemos observar com clareza a tendência geral (gráfico 2), a sazonalidade (gráfico 3) e os resíduos (gráfico 4).

    Ao trabalhar com séries temporais - dependendo do modelo selecionado - é importate entender se a série é estacionaria ou nao-estacionaria. O teste Augmented Dickey-Fuller
    nos ajuda a entender se o conjunto em questao é ou nao estacionarios:
    """
    f"""
    Aplicando-se o teste de Dickey-Fuller, temos um valor de P-value = {p_value}. Dessa maneira, não podemos rejeitar a hipótese nula, o que significa
    que temos uma série não estacionária.
    """
    """
    Abaixo temos uma representação visual da média movel em relaçao ao valores:
    """
    
    ma = df.rolling(260).mean()

    f, ax = plt.subplots()
    df.plot(ax=ax, legend=False)
    ma.plot(ax=ax, legend = False, color = 'r')
    plt.xlabel('Ano')
    plt.ylabel('Valor (US$)')

    plt.tight_layout()
    st.pyplot(plt)

    """
    Considerando que estamos trabalhando com uma série atualmente não-estacionaria, a primeira coisa que precisamos fazer é transforma-la. 
    Para isso aplicamos a primeira derivada:
    """
    df_diff = df.diff(1)
    ma_diff = df_diff.rolling(247).mean()

    std_diff = df_diff.rolling(247).std()

    f, ax = plt.subplots()
    df_diff.plot(ax=ax, legend=False)
    ma_diff.plot(ax=ax, color='r', legend=False)
    std_diff.plot(ax=ax, color='g', legend=False)
    plt.xlabel('Ano')
    plt.tight_layout()
    st.pyplot(plt)

    X_diff = df_diff.y.dropna().values
    result_diff = adfuller(X_diff)

    f"""
    Nesse novo formato, aplicamos novamente o teste de Dickey-Fuller, e chegamos a um valor de P-value muito próximo a 
    0, de modo que podemos rejeitar a hipótese nula, e considerar que a série agora é estacionária.
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
