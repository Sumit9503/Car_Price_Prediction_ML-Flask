from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Define the directory and filenames
model_dir = 'saved_models'
model_filename = os.path.join(model_dir, 'tuned_xgb_regressor_model.pkl')
scaler_filename = os.path.join(model_dir, 'scaler.pkl')

# Load model and scaler
try:
    model = joblib.load(model_filename)
    scaler = joblib.load(scaler_filename)
    print("Model and scaler loaded successfully!")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

# Define expected features and numerical columns (42 features total)
expected_features = [
    'symboling', 'doornumber', 'wheelbase', 'carlength', 'carwidth', 'carheight',
    'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower',
    'peakrpm', 'fueleconomy', 'fueltype_gas', 'aspiration_turbo', 'carbody_convertible',
    'carbody_hardtop', 'carbody_hatchback', 'carbody_sedan', 'drivewheel_fwd',
    'drivewheel_rwd', 'enginelocation_rear', 'enginetype_dohc', 'enginetype_dohcv',
    'enginetype_l', 'enginetype_ohc', 'enginetype_ohcf', 'enginetype_ohcv',
    'enginetype_rotor', 'cylindernumber_eight', 'cylindernumber_five',
    'cylindernumber_four', 'cylindernumber_six', 'cylindernumber_three',
    'cylindernumber_twelve', 'fuelsystem_1bbl', 'fuelsystem_2bbl', 'fuelsystem_4bbl',
    'fuelsystem_idi', 'fuelsystem_mfi', 'fuelsystem_mpfi', 'fuelsystem_spdi'
]

numerical_cols_for_scaling = [
    'wheelbase', 'curbweight', 'enginesize', 'boreratio', 'horsepower', 'fueleconomy',
    'carlength', 'carwidth', 'doornumber'
]

def generate_charts(predicted_price, input_specs):
    charts = {}

    # Price Comparison Chart (Bar Chart)
    charts['price_comparison'] = {
        'data': [
            {
                'type': 'bar',
                'x': ['Predicted Price', 'Average Sedan', 'Average Hatchback'],
                'y': [predicted_price, 15000, 12000],
                'marker': {
                    'color': ['#06d6a0', '#8b5cf6', '#ffd23f']
                }
            }
        ],
        'layout': {
            'title': {'text': 'Price Comparison', 'font': {'color': '#b0b3c7'}},
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'font': {'color': '#b0b3c7'},
            'xaxis': {'title': 'Category'},
            'yaxis': {'title': 'Price ($)'}
        }
    }

    # Performance Scatter Chart
    charts['performance_scatter'] = {
        'data': [
            {
                'type': 'scatter',
                'mode': 'markers',
                'x': [input_specs['horsepower']],
                'y': [input_specs['fueleconomy']],
                'marker': {
                    'size': 15,
                    'color': '#06d6a0'
                },
                'name': 'Your Car'
            },
            {
                'type': 'scatter',
                'mode': 'markers',
                'x': [100, 150, 200],
                'y': [30, 25, 20],
                'marker': {
                    'size': 10,
                    'color': '#8b5cf6'
                },
                'name': 'Other Cars'
            }
        ],
        'layout': {
            'title': {'text': 'Horsepower vs Fuel Economy', 'font': {'color': '#b0b3c7'}},
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'font': {'color': '#b0b3c7'},
            'xaxis': {'title': 'Horsepower'},
            'yaxis': {'title': 'Fuel Economy (MPG)'}
        }
    }

    # Radar Chart
    charts['radar'] = {
        'data': [
            {
                'type': 'scatterpolar',
                'r': [
                    input_specs['wheelbase'] / 120.9,
                    input_specs['carlength'] / 208.1,
                    input_specs['carwidth'] / 72.0,
                    input_specs['horsepower'] / 262,
                    input_specs['fueleconomy'] / 50
                ],
                'theta': ['Wheelbase', 'Car Length', 'Car Width', 'Horsepower', 'Fuel Economy'],
                'fill': 'toself',
                'line': {'color': '#06d6a0'}
            }
        ],
        'layout': {
            'title': {'text': 'Car Specs Radar', 'font': {'color': '#b0b3c7'}},
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'font': {'color': '#b0b3c7'},
            'polar': {
                'bgcolor': 'rgba(0,0,0,0)',
                'radialaxis': {'visible': True, 'range': [0, 1]}
            }
        }
    }

    # Gauge Chart (Performance Score)
    performance_score = (input_specs['horsepower'] / 262) * 100
    charts['gauge'] = {
        'data': [
            {
                'type': 'indicator',
                'mode': 'gauge+number',
                'value': performance_score,
                'title': {'text': 'Performance Score', 'font': {'color': '#b0b3c7'}},
                'gauge': {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#06d6a0'},
                    'bgcolor': 'rgba(0,0,0,0)',
                    'bordercolor': '#b0b3c7'
                }
            }
        ],
        'layout': {
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'font': {'color': '#b0b3c7'}
        }
    }

    return charts

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model or scaler not loaded properly'}), 500
    
    try:
        # Get form data
        input_data = {
            'symboling': float(request.form.get('symboling', 1)),
            'wheelbase': float(request.form.get('wheelbase', 95.0)),
            'carlength': float(request.form.get('carlength', 175.0)),
            'carwidth': float(request.form.get('carwidth', 65.0)),
            'carheight': float(request.form.get('carheight', 54.0)),
            'curbweight': float(request.form.get('curbweight', 2500)),
            'enginesize': float(request.form.get('enginesize', 150)),
            'boreratio': float(request.form.get('boreratio', 3.3)),
            'stroke': float(request.form.get('stroke', 3.2)),
            'compressionratio': float(request.form.get('compressionratio', 9.0)),
            'horsepower': float(request.form.get('horsepower', 150)),
            'peakrpm': float(request.form.get('peakrpm', 5000)),
            'citympg': float(request.form.get('citympg', 25)),
            'highwaympg': float(request.form.get('highwaympg', 30)),
            'fueltype': request.form.get('fueltype', 'gas'),
            'aspiration': request.form.get('aspiration', 'std'),
            'doornumber': int(request.form.get('doornumber', 4)),
            'carbody': request.form.get('carbody', 'sedan'),
            'drivewheel': request.form.get('drivewheel', 'fwd'),
            'enginetype': request.form.get('enginetype', 'ohc'),
            'cylindernumber': request.form.get('cylindernumber', 'four'),
            'enginelocation': 'front',
            'fuelsystem': 'mpfi'
        }
        
        # Save input specs for display
        input_specs = input_data.copy()
        input_specs['drivewheel'] = input_specs['drivewheel'].upper()
        input_specs['fueltype'] = input_specs['fueltype'].capitalize()
        input_specs['aspiration'] = input_specs['aspiration'].capitalize()
        input_specs['carbody'] = input_specs['carbody'].capitalize()
        input_specs['enginetype'] = input_specs['enginetype'].upper()
        input_specs.pop('symboling', None)
        input_specs.pop('stroke', None)
        input_specs.pop('compressionratio', None)
        input_specs.pop('peakrpm', None)
        input_specs.pop('citympg', None)
        input_specs.pop('highwaympg', None)
        input_specs.pop('enginelocation', None)
        input_specs.pop('fuelsystem', None)
        input_specs['fueleconomy'] = (0.55 * input_data['citympg'] + 0.45 * input_data['highwaympg'])
        
        # Process input data
        input_data_df = pd.DataFrame([input_data])
        
        # Compute fueleconomy and drop citympg, highwaympg
        input_data_df['fueleconomy'] = (0.55 * input_data_df['citympg']) + (0.45 * input_data_df['highwaympg'])
        input_data_df = input_data_df.drop(['citympg', 'highwaympg'], axis=1)
        
        # One-hot encoding
        categorical_cols = ['fueltype', 'aspiration', 'carbody', 'drivewheel', 
                           'enginetype', 'cylindernumber', 'enginelocation', 'fuelsystem']
        input_data_processed = pd.get_dummies(input_data_df, columns=categorical_cols, drop_first=True, dtype='int')
        
        # Align columns with expected features
        missing_cols = set(expected_features) - set(input_data_processed.columns)
        for c in missing_cols:
            input_data_processed[c] = 0
        
        extra_cols = set(input_data_processed.columns) - set(expected_features)
        input_data_processed = input_data_processed.drop(list(extra_cols), axis=1)
        input_data_processed = input_data_processed[expected_features]
        
        # Scale numerical features
        numerical_cols_in_input = [col for col in numerical_cols_for_scaling if col in input_data_processed.columns]
        if numerical_cols_in_input:
            input_data_processed[numerical_cols_in_input] = scaler.transform(input_data_processed[numerical_cols_in_input])
        
        # Make prediction
        log_predicted_price = float(model.predict(input_data_processed.values)[0])
        predicted_price = np.expm1(log_predicted_price)  # Reverse log1p transformation
        
        # Generate charts
        charts = generate_charts(predicted_price, input_specs)
        
        # Return JSON response
        response_data = {
            'predicted_price': round(predicted_price, 2),
            'input_specs': input_specs,
            'charts': charts
        }
        return jsonify(response_data), 200
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)