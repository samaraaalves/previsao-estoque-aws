import pandas as pd
from sklearn.linear_model import LinearRegression


dados = {
    'mes': [1, 2, 3, 4, 5, 6],
    'vendas': [100, 120, 130, 150, 170, 200]
}

df = pd.DataFrame(dados)


X = df[['mes']]
y = df['vendas']


modelo = LinearRegression()
modelo.fit(X, y)


meses_futuros = pd.DataFrame({'mes': [7, 8, 9]})
previsoes = modelo.predict(meses_futuros)


resultado = pd.DataFrame({
    'mes': [7, 8, 9],
    'previsao_vendas': previsoes
})

print(resultado)