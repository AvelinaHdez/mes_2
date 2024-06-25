from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar el modelo y el escalador
model = joblib.load('modelo.pkl')
scaler = joblib.load('escalador.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    total = None
    if request.method == 'POST':
        # Obtener los datos del formulario
        horas = float(request.form['horas'])
        tasa = float(request.form['tasa'])
        ciudad = float(request.form['ciudad'])
        compania = float(request.form['compania'])

        # Crear el DataFrame
        data = pd.DataFrame([[horas, tasa, ciudad, compania]], columns=['MonthlyHours', 'TariffRate', 'City', 'Company'])

        # Escalar los datos
        data_scaled = scaler.transform(data)

        # Hacer la predicción
        prediction = model.predict(data_scaled)
        total = prediction[0]

    return render_template('index.html', total=total)

if __name__ == '__main__':
    app.run(debug=True)
