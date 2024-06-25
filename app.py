from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('modelo.pkl')
scaler = joblib.load('escalador.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            pm10 = float(request.form['pm10'])
            co = float(request.form['co'])
            nh3 = float(request.form['nh3'])
            o3 = float(request.form['o3'])
            
            # Create DataFrame
            data = pd.DataFrame([[pm10, co, nh3, o3]], columns=['pm10', 'co', 'nh3', 'o3'])
            
            # Scale the data
            data_scaled = scaler.transform(data)
            
            # Predict
            pm2_5 = model.predict(data_scaled)[0]
            
            # Determine air quality message
            if pm2_5 <= 12:
                message = f'El PM2.5 es de {pm2_5:.2f}, el aire es bueno.'
            elif pm2_5 <= 35.4:
                message = f'El PM2.5 es de {pm2_5:.2f}, el aire es moderado.'
            elif pm2_5 <= 55.4:
                message = f'El PM2.5 es de {pm2_5:.2f}, el aire es dañino para grupos sensibles.'
            elif pm2_5 <= 150.4:
                message = f'El PM2.5 es de {pm2_5:.2f}, el aire es dañino.'
            elif pm2_5 <= 250.4:
                message = f'El PM2.5 es de {pm2_5:.2f}, el aire es muy dañino.'
            else:
                message = f'El PM2.5 es de {pm2_5:.2f}, el aire es peligroso.'
            
            return render_template('index.html', message=message)
        
        except ValueError:
            return render_template('index.html', message='Por favor, ingrese valores válidos.')
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
