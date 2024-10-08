import pandas as pd
import yfinance as yf
import talib
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np

# Escolha do ticker da ação
ticker = 'PETR4.SA'

# Coleta de dados da ação
data = yf.download(ticker, start='2016-01-01', end='2024-09-30')

# Calcular indicadores técnicos usando TA-Lib
data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
data['SMA'] = talib.SMA(data['Close'], timeperiod=30)

# Identificar padrões de CandleStick
data['Doji'] = talib.CDLDOJI(data['Open'], data['High'], data['Low'], data['Close'])
data['Hammer'] = talib.CDLHAMMER(data['Open'], data['High'], data['Low'], data['Close'])
data['Engulfing'] = talib.CDLENGULFING(data['Open'], data['High'], data['Low'], data['Close'])

# Função para determinar recomendação de compra ou venda
def get_recommendation(row):
    if row['Hammer'] != 0 and row['RSI'] < 30:
        return 'Compra'
    elif row['Doji'] != 0 and row['RSI'] > 70:
        return 'Venda'
    elif row['Engulfing'] == 100:  # Bullish Engulfing
        return 'Compra (Bullish Engulfing)'
    elif row['Engulfing'] == -100:  # Bearish Engulfing
        return 'Venda (Bearish Engulfing)'
    else:
        return 'Nenhuma'

# Aplicar a função ao DataFrame
data['Recomendacao'] = data.apply(get_recommendation, axis=1)

# Adicionando a coluna de variação de preço para calcular se as recomendações foram corretas
data['Preco_Futuro'] = data['Close'].shift(-1)  # Preço do dia seguinte
data['Variação'] = data['Preco_Futuro'] - data['Close']

# Verificando acurácia: compra quando o preço sobe, venda quando o preço desce
data['Recomendacao_Correta'] = data.apply(lambda row: 
                                          (row['Recomendacao'].startswith('Compra') and row['Variação'] > 0) or
                                          (row['Recomendacao'].startswith('Venda') and row['Variação'] < 0), axis=1)

# Filtrando apenas sinais de compra e venda para calcular a acurácia
relevant_data = data[data['Recomendacao'].str.contains('Compra|Venda')]

# Cálculo da acurácia
accuracy = relevant_data['Recomendacao_Correta'].mean()

# Cálculo do RMSE para os preços previstos (sinais de compra/venda)
rmse = np.sqrt(mean_squared_error(relevant_data['Close'], relevant_data['Preco_Futuro']))

# Exibir acurácia e RMSE
print(f"Acurácia das Recomendações: {accuracy:.2f}")
print(f"RMSE das Previsões de Preço: {rmse:.2f}")

# Filtrar recomendações de compra e venda
buy_signals = data[data['Recomendacao'].str.contains('Compra')]
sell_signals = data[data['Recomendacao'].str.contains('Venda')]
bullish_engulfing = data[data['Engulfing'] == 100]
bearish_engulfing = data[data['Engulfing'] == -100]

# Criar o gráfico de CandleStick
fig = go.Figure(data=[go.Candlestick(x=data.index,
                                     open=data['Open'],
                                     high=data['High'],
                                     low=data['Low'],
                                     close=data['Close'],
                                     name='Candlestick')])

# Adicionar a média móvel simples (SMA) ao gráfico com a cor amarela
fig.add_trace(go.Scatter(x=data.index, y=data['SMA'], mode='lines', name='Média Móvel de 30 dias', 
                         line=dict(color='yellow')))

# Adicionar sinais de compra (azul) ao gráfico
fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', 
                         marker=dict(symbol='triangle-up', color='blue', size=10), 
                         name='Sinal de Compra'))

# Adicionar sinais de venda (roxo) ao gráfico
fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', 
                         marker=dict(symbol='triangle-down', color='purple', size=10), 
                         name='Sinal de Venda'))

# Adicionar Bullish Engulfing ao gráfico com marcadores verdes
fig.add_trace(go.Scatter(x=bullish_engulfing.index, y=bullish_engulfing['Close'], mode='markers', 
                         marker=dict(symbol='triangle-up', color='green', size=12), 
                         name='Bullish Engulfing'))

# Adicionar Bearish Engulfing ao gráfico com marcadores vermelhos
fig.add_trace(go.Scatter(x=bearish_engulfing.index, y=bearish_engulfing['Close'], mode='markers', 
                         marker=dict(symbol='triangle-down', color='red', size=12), 
                         name='Bearish Engulfing'))

# Configurar o layout do gráfico
fig.update_layout(
    title=f'Gráfico de Candlestick para {ticker} com Padrões Bullish e Bearish Engulfing',
    xaxis_title='Data',
    yaxis_title='Preço',
    xaxis_rangeslider_visible=False,
    template='plotly_dark',
    legend=dict(
        title="Padrões Candlestick",
        orientation="h",  # Legenda horizontal
        x=0.5, xanchor='center',  # Centraliza a legenda
        y=-0.2  # Posição abaixo do gráfico
    )
)

# Exibir o gráfico interativo
fig.show()
