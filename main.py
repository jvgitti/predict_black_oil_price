from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import numpy as np
import joblib

import streamlit as st

st.title('Análise para predição dos valores do barril de petróleo')

tab_0, tab_1 = st.tabs(['Análise Exploratória', 'Modelo'])

df = pd.read_csv('oil_price.csv', sep='\t', names=['ds', 'y'])

df['ds'] = pd.to_datetime(df['ds'])
df['y'] = df['y'].str.replace(',', '.').astype(float)
df = df.set_index('ds')

ad_fuller = adfuller(df.y.values)
p_value = ad_fuller[1]


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


def calcula_wmape(y_true, y_pred):
    return np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()


df = df.reset_index('ds')
df['unique_id'] = 0
df_treino = df[df.ds < '2022-12-22']
df_valid = df[df.ds >= '2022-12-22']

# from statsforecast import StatsForecast
# from statsforecast.models import Naive, MSTL
#
# model = StatsForecast(models=[MSTL(season_length=[247, 22, 5], trend_forecaster=Naive())], freq='B', n_jobs=-1)
# model.fit(df_treino)
# joblib.dump(model, 'model.joblib')

with tab_1:
    """
    Avaliamos e testamos diversos modelos para a predição dos valores de fechamento, e o que apresentou o melhor resultado foi o modelo Naive,
    juntamente com o MSTL, pois conseguimos passar diversas sazonalidades para o modelo em questão.
    
    Estudando as variáveis que podem interferir no nosso modelo, optamos por treinar o modelo com uma sazonalidade anual, mensal e semanal.
    Como período, utilizamos o período diário, considerando apenas os dias úteis, em que a bolsa està aberta.
    
    Utilizamos todo o período de dados para treinar o modelo (de 2003 à 2023).
    
    Abaixo segue o resultado do modelo, com os dados reais de 2019 até os dias atuais, e a predição realizada a partir de Agosto/2022.
    
    Com a barra, pode-se escolher o período de predição desejado, com o máximo de 260 dias úteis, aproximadamente 1 ano no total.
    """

    h = st.slider("Selecione um valor entre 50 e 260, para a predição:", 50, 260, value=260)
    model = joblib.load('model.joblib')
    forecast_df = model.predict(h=h, level=[90])
    forecast_df = forecast_df.reset_index().merge(df_valid, on=['ds', 'unique_id'], how='left')
    forecast_df = forecast_df.dropna()

    wmape = calcula_wmape(forecast_df['y'].values, forecast_df['MSTL'].values)

    import plotly.express as px
    import plotly.graph_objects as go

    fig = px.line(df_treino[df_treino.ds >= '2019-05-01'], x='ds', y='y', line_shape='linear')
    fig.add_scatter(x=forecast_df['ds'], y=forecast_df['y'], mode='lines', name='Real')
    fig.add_scatter(x=forecast_df['ds'], y=forecast_df['MSTL'], mode='lines', name='Predito')
    fig.add_scatter(x=forecast_df['ds'], y=forecast_df['MSTL-lo-90'], mode='lines', name='Intervalo de confiança')
    fig.add_scatter(x=forecast_df['ds'], y=forecast_df['MSTL-hi-90'], mode='lines', name='Intervalo de confiança', showlegend=False)

    fig.update_traces(line=dict(color='blue'), selector=dict(name='Real'))
    fig.update_traces(line=dict(color='green'), selector=dict(name='Predito'))
    fig.update_traces(line=dict(color='aquamarine'), selector=dict(name='Intervalo de confiança'))

    fig.update_layout(
        title='Preço do Petróleo Bruto - Predição do valor',
        xaxis_title='Ano',
        yaxis_title='Valor (US$)',
        legend=dict(
            x=0,
            y=-0.2,
            orientation="h",
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    f"""
    WMAPE para o período fornecido: {wmape:.2%}
    """

    f"""
    Variando-se o período de predição, percebe-se que o modelo varia o valor do WMAPE, apresentando um resultado interessante no longo prazo.
    """

