import requests
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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
ticker = 'BBAS3.SA'
start_date = '2014-01-01'
end_date = '2024-09-30'

data = get_stock_data(ticker, start=start_date, end=end_date)

# Remover a coluna 'Date' (mantendo apenas o índice)
data.reset_index(inplace=True)

# Adicionando a taxa Selic ao DataFrame (combinando por data)
data = pd.merge(data, selic_df, on='Date', how='left')

# Preenchendo valores nulos de Selic com o último valor disponível
data['Selic Rate'].fillna(method='ffill', inplace=True)

# Features (usando CandleStick e taxa Selic)
X = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Selic Rate']]
y = np.where(data['Close'] > data['Open'], 1, 0)  # Definindo padrões de CandleStick como base

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelo de Machine Learning (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Previsões
y_pred = model.predict(X_test)

# Avaliação do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy do modelo: {accuracy:.2f}")

# Função de recomendação de compra ou venda
def investment_recommendation(model, X):
    prediction = model.predict(X)
    if prediction == 1:
        return "Recomenda-se COMPRA."
    else:
        return "Recomenda-se VENDA."

# Teste com novos dados para gerar a recomendação
sample_data = X_test.iloc[0:1]  # Usando a primeira linha de teste como exemplo
recommendation = investment_recommendation(model, sample_data)

# Exibindo a recomendação de compra ou venda
print(f"Recomendacao baseada nos dados recentes: {recommendation}")

# Criando um gráfico interativo com Plotly

# Criando a figura com eixos duplos
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

# Ajustando os títulos e eixos
fig.update_layout(
    title_text="Candlestick e Taxa Selic (Interativo)",
    xaxis_title="Data",
    yaxis_title="Preço da Ação",
    legend_title="Indicadores",
    hovermode="x unified"
)

# Configurando o eixo y secundário para a taxa Selic
fig.update_yaxes(title_text="Taxa Selic (%)", secondary_y=True)

# Exibindo o gráfico interativo no navegador
pyo.plot(fig)

# Exibir recomendação também no console
print(f"Recomendação Final: {recommendation}")
