import pandas as pd
import yfinance as yf
import talib
import plotly.graph_objects as go

# Coleta de dados da ação, por exemplo AAPL
data = yf.download('PETR4.SA', start='2016-01-01', end='2024-09-30')

# Calcular indicadores técnicos usando TA-Lib
data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
data['SMA'] = talib.SMA(data['Close'], timeperiod=30)

# Identificar padrões de CandleStick
data['Doji'] = talib.CDLDOJI(data['Open'], data['High'], data['Low'], data['Close'])
data['Hammer'] = talib.CDLHAMMER(data['Open'], data['High'], data['Low'], data['Close'])

# Função para determinar recomendação de compra ou venda
def get_recommendation(row):
    if row['Hammer'] != 0 and row['RSI'] < 30:
        return 'Compra'
    elif row['Doji'] != 0 and row['RSI'] > 70:
        return 'Venda'
    else:
        return 'Nenhuma'

# Aplicar a função ao DataFrame
data['Recomendacao'] = data.apply(get_recommendation, axis=1)

# Filtrar recomendações de compra e venda
buy_signals = data[data['Recomendacao'] == 'Compra']
sell_signals = data[data['Recomendacao'] == 'Venda']

# Criar o gráfico de CandleStick
fig = go.Figure(data=[go.Candlestick(x=data.index,
                                     open=data['Open'],
                                     high=data['High'],
                                     low=data['Low'],
                                     close=data['Close'],
                                     name='Candlestick')])

# Adicionar a média móvel simples (SMA) ao gráfico
fig.add_trace(go.Scatter(x=data.index, y=data['SMA'], mode='lines', name='Média Móvel de 30 dias'))

# Adicionar sinais de compra (verde) ao gráfico
fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', 
                         marker=dict(symbol='triangle-up', color='blue', size=10), 
                         name='Sinal de Compra'))

# Adicionar sinais de venda (vermelho) ao gráfico
fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', 
                         marker=dict(symbol='triangle-down', color='purple', size=10), 
                         name='Sinal de Venda'))

# Configurar o layout do gráfico
fig.update_layout(
    title='Gráfico de Candlestick com Recomendações de Compra e Venda',
    xaxis_title='Data',
    yaxis_title='Preço',
    xaxis_rangeslider_visible=False,
    template='plotly_dark'
)

# Exibir o gráfico interativo
fig.show()
