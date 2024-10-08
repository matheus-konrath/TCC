import requests
import pandas as pd
import numpy as np
import yfinance as yf
import talib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

# URL do JSON com os dados da taxa Selic
json_url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.1178/dados?formato=json&dataInicial=01/01/2014&dataFinal=30/09/2024"

# Fazendo a requisição para obter o JSON
response = requests.get(json_url)
selic_data = response.json()

# Convertendo os dados da Selic para DataFrame
selic_df = pd.DataFrame(selic_data)
selic_df['data'] = pd.to_datetime(selic_df['data'], format='%d/%m/%Y')
selic_df['valor'] = selic_df['valor'].astype(float)
selic_df.columns = ['Date', 'Selic Rate']

# Função para coletar dados de ações e indicadores
def get_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

# Coletando dados de uma ação
ticker = 'ITUB4.SA'
start_date = '2014-01-01'
end_date = '2024-09-30'

data = get_stock_data(ticker, start=start_date, end=end_date)

# Remover a coluna 'Date' (mantendo apenas o índice)
data.reset_index(inplace=True)

# Adicionando a taxa Selic ao DataFrame (combinando por data)
data = pd.merge(data, selic_df, on='Date', how='left')

# Preenchendo valores nulos de Selic com o último valor disponível
data['Selic Rate'].fillna(method='ffill', inplace=True)

# Aplicando TA-Lib para identificar padrões de Candlestick
# Exemplos de padrões de candlestick
data['CDL_ENGULFING'] = talib.CDLENGULFING(data['Open'], data['High'], data['Low'], data['Close'])
data['CDL_DOJI'] = talib.CDLDOJI(data['Open'], data['High'], data['Low'], data['Close'])

# Recomendação com base nos padrões
def candlestick_recommendation(row):
    if row['CDL_ENGULFING'] > 0 or row['CDL_DOJI'] > 0:
        return "Compra"
    elif row['CDL_ENGULFING'] < 0 or row['CDL_DOJI'] < 0:
        return "Venda"
    else:
        return "Manter"

# Aplicando a recomendação para cada linha de dados
data['Recommendation'] = data.apply(candlestick_recommendation, axis=1)

# Criando o gráfico com Plotly
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Gráfico Candlestick
fig.add_trace(go.Candlestick(x=data['Date'],
                             open=data['Open'],
                             high=data['High'],
                             low=data['Low'],
                             close=data['Close'],
                             name='Preço Ação'),
              secondary_y=False)

# Gráfico da taxa Selic (em linha)
fig.add_trace(go.Scatter(x=data['Date'], y=data['Selic Rate'],
                         mode='lines', name='Taxa Selic', line=dict(color='green')),
              secondary_y=True)

# Adicionando as recomendações ao gráfico
buy_signals = data['Date'][data['Recommendation'] == "Compra"]
sell_signals = data['Date'][data['Recommendation'] == "Venda"]

# Marcando os pontos de compra (seta verde)
fig.add_trace(go.Scatter(x=buy_signals, 
                         y=data['Close'][data['Recommendation'] == "Compra"],
                         mode='markers', 
                         marker=dict(color='green', symbol='triangle-up', size=10),
                         name='Sinal de Compra'))

# Marcando os pontos de venda (seta vermelha)
fig.add_trace(go.Scatter(x=sell_signals, 
                         y=data['Close'][data['Recommendation'] == "Venda"],
                         mode='markers', 
                         marker=dict(color='red', symbol='triangle-down', size=10),
                         name='Sinal de Venda'))

# Ajustando os títulos e eixos
fig.update_layout(
    title_text="Candlestick e Taxa Selic com Recomendações de Compra e Venda (TA-Lib)",
    xaxis_title="Data",
    yaxis_title="Preço da Ação",
    legend_title="Indicadores",
    hovermode="x unified"
)

# Configurando o eixo y secundário para a taxa Selic
fig.update_yaxes(title_text="Taxa Selic (%)", secondary_y=True)

# Exibindo o gráfico interativo no navegador
pyo.plot(fig)

# Exibir as primeiras recomendações
print(data[['Date', 'Close', 'Recommendation']].head())
