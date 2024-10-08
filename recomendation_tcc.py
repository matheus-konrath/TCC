import pandas as pd
import yfinance as yf
import talib
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
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

# Preparar as features e o target removendo valores ausentes
features = ['RSI', 'SMA']
X = data[features].dropna()  # Remove os NaN nas features
y = data['Close'].loc[X.index]  # Sincroniza o y com o índice de X, removendo os NaN

# Dividir os dados para treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Seleção de modelos
models = {
    'LinearRegression': LinearRegression(),
    'DecisionTree': DecisionTreeRegressor(),
    'RandomForest': RandomForestRegressor(),
    'NeuralNetwork': MLPRegressor(max_iter=1000)
}

# Avaliação de modelos e ajuste de hiperparâmetros (Grid Search)
best_model = None
best_rmse = np.inf
for model_name, model in models.items():
    if model_name == 'RandomForest':  # Exemplo de grid search para Random Forest
        param_grid = {'n_estimators': [10, 50, 100], 'max_depth': [5, 10, None]}
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        best_model_params = grid_search.best_params_
        model = grid_search.best_estimator_
    else:
        model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model_name

    print(f"Modelo: {model_name}, RMSE: {rmse}")

print(f"Melhor modelo: {best_model} com RMSE: {best_rmse}")

# Sistema de Recomendação de Ações
def recommendation_system(predictions, profile):
    if profile == 'conservador':
        return 'Aguarde, mercado volatil'
    elif profile == 'moderado':
        return 'Considere acoes de baixo risco'
    elif profile == 'arrojado':
        return 'Considere acoes de crescimento'
    else:
        return 'Perfil não encontrado'

# Exemplo de recomendação com base no perfil de risco
perfil_investidor = 'moderado'  # Exemplo: conservador, moderado, agressivo
recomendacao = recommendation_system(y_pred, perfil_investidor)
print(f"Recomendacao para o perfil {perfil_investidor}: {recomendacao}")

# Gerar previsões no conjunto de dados completo (usando apenas linhas válidas)
data['Predictions'] = np.nan  # Cria uma coluna de previsões com NaN inicialmente
data.loc[X.index, 'Predictions'] = model.predict(X)  # Preenche com previsões nas linhas válidas

# Transformar o problema em classificação: prever se o preço vai subir ou cair
y_test_class = (y_test.shift(-1) > y_test).astype(int)  # 1 se o preço sobe, 0 se o preço cai
y_pred_class = (data.loc[X_test.index, 'Predictions'].shift(-1) > data.loc[X_test.index, 'Predictions']).astype(int)

# Remover valores NaN gerados pela mudança (shift)
y_test_class = y_test_class.dropna()
y_pred_class = y_pred_class.dropna()

# Calcular a acurácia
accuracy = accuracy_score(y_test_class, y_pred_class)
print(f"Acuracia do modelo: {accuracy}")

# Backtesting (Teste retroativo com dados históricos)
data['Returns'] = data['Close'].pct_change()  # Retornos diários

# Gráfico de CandleStick com Médias Móveis e Sinais de Compra/Venda
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
buy_signals = data[data['Recomendacao'].str.contains('Compra')]
fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', 
                         marker=dict(symbol='triangle-up', color='blue', size=10), 
                         name='Sinal de Compra'))

# Adicionar sinais de venda (roxo) ao gráfico
sell_signals = data[data['Recomendacao'].str.contains('Venda')]
fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', 
                         marker=dict(symbol='triangle-down', color='purple', size=10), 
                         name='Sinal de Venda'))

# Configurar o layout do gráfico
fig.update_layout(
    title=f'Gráfico de Candlestick para {ticker} com Seleção de Modelos e Recomendações',
    xaxis_title='Data',
    yaxis_title='Preço',
    xaxis_rangeslider_visible=False,
    template='plotly_dark',
    legend=dict(
        title="Padrões Candlestick",
        orientation="h",
        x=0.5, xanchor='center',
        y=-0.2
    )
)

# Exibir o gráfico interativo
fig.show()
