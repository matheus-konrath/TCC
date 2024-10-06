import yfinance as yf
import talib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import plotly.graph_objs as go

# Baixar dados de ações e petróleo (incluindo o BRENT)
tickers = ['PETR4.SA', 'CL=F', 'BZ=F']
data = yf.download(tickers, start='2015-01-01', end='2024-09-30')['Adj Close']

# Preencher dados ausentes
data = data.ffill()

# Normalizar os dados para comparação em uma mesma escala
data['PETR4_normalized'] = data['PETR4.SA'] / data['PETR4.SA'].max()
data['Brent_normalized'] = data['BZ=F'] / data['BZ=F'].max()

# Criar features técnicas usando TA-Lib (como padrões de CandleStick)
data['RSI'] = talib.RSI(data['PETR4.SA'], timeperiod=14)
data['EMA'] = talib.EMA(data['PETR4.SA'], timeperiod=30)

# Gerar um padrão de alta/breve reversão usando um padrão de CandleStick (Exemplo: Hammer)
data['Candle_Pattern'] = talib.CDLHAMMER(data['PETR4.SA'], data['PETR4.SA'], data['PETR4.SA'], data['PETR4.SA'])

# Remover linhas com valores NaN (causadas pelo cálculo dos indicadores)
data = data.dropna()

# Variável alvo (1 = Compra, 0 = Venda) baseada em variações de preço
data['Target'] = np.where(data['PETR4.SA'].shift(-1) > data['PETR4.SA'], 1, 0)

# Separar dados em treino e teste
X = data[['RSI', 'EMA', 'Candle_Pattern']]
y = data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Treinar um modelo Random Forest
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Avaliação do modelo
accuracy = model.score(X_test, y_test)
print(f"Acurácia: {accuracy}")

# Prever ações futuras
predictions = model.predict(X_test)

# Criar uma nova coluna 'Signals' apenas para os dados de teste
data.loc[X_test.index, 'Signals'] = predictions

# Função para reduzir o número de sinais mantendo uma distância mínima entre eles
def reduce_signals(signals, min_distance=10):
    reduced_signals = []
    last_signal = -min_distance  # Inicializando o último sinal como fora do alcance
    for i in range(len(signals)):
        if signals[i] and (i - last_signal) >= min_distance:
            reduced_signals.append(i)
            last_signal = i
    return reduced_signals

# Reduzir o número de sinais de compra e venda
buy_signals = reduce_signals(data['Signals'] == 1, min_distance=10)
sell_signals = reduce_signals(data['Signals'] == 0, min_distance=10)

# Gráfico interativo com Plotly
fig = go.Figure()

# Adicionar os preços normalizados da PETR4
fig.add_trace(go.Scatter(x=data.index, y=data['PETR4_normalized'], mode='lines', name='PETR4 (Normalizado)', line=dict(color='blue')))

# Adicionar os preços normalizados do Brent (BZ=F)
fig.add_trace(go.Scatter(x=data.index, y=data['Brent_normalized'], mode='lines', name='Brent Crude (BZ=F) (Normalizado)', line=dict(color='orange')))

# Adicionar os pontos de compra reduzidos
fig.add_trace(go.Scatter(x=data.iloc[buy_signals].index, y=data['PETR4_normalized'].iloc[buy_signals], mode='markers', name='Sinal de Compra', marker=dict(color='green', size=10, symbol='triangle-up')))

# Adicionar os pontos de venda reduzidos
fig.add_trace(go.Scatter(x=data.iloc[sell_signals].index, y=data['PETR4_normalized'].iloc[sell_signals], mode='markers', name='Sinal de Venda', marker=dict(color='red', size=10, symbol='triangle-down')))

# Customizar o layout do gráfico
fig.update_layout(
    title='Sinais de Compra e Venda para PETR4 e Comparação com o Brent (Normalizado)',
    xaxis_title='Data',
    yaxis_title='Preço (Normalizado)',
    hovermode="x unified"
)

# Mostrar o gráfico interativo
fig.show()
