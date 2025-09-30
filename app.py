from flask import Flask, request, jsonify, render_template

# Initialize the Flask application
app = Flask(__name__)


# --- Model Simulation ---
# In a real-world scenario, you would load a trained model here,
# for example:
# import pickle
# with open('model.pkl', 'rb') as file:
#     model = pickle.load(file)
#
# For this example, we will simulate the model with a simple formula:
# Salary = (2000 * Years of Experience) + 30000
# This hardcoded logic serves the same purpose of demonstrating the backend.
def predict_salary(years_of_experience):
    """
    Simulates a linear regression model prediction.
    """
    if not isinstance(years_of_experience, (int, float)):
        raise TypeError("Input must be a number.")

    # Simple linear equation to predict salary
    predicted_salary = (9339.08172382 * years_of_experience) + 25918.438334893202

    return predicted_salary


# --- Flask Routes ---

@app.route('/')
def home():
    """
    Renders the main HTML page.
    This assumes your 'index.html' file is in a 'templates' directory.
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction request from the HTML page.
    """
    # Get the JSON data from the request body
    data = request.get_json(force=True)

    # Extract the 'experience' value from the JSON data
    years_of_experience = data.get('experience')

    # Basic input validation
    if years_of_experience is None:
        return jsonify({'error': 'No experience value provided'}), 400

    try:
        # Use the simulated model to get the prediction
        prediction = predict_salary(years_of_experience)

        # Return the prediction as a JSON response
        # The result is formatted to be a currency value
        return jsonify({
            'predicted_salary': round(prediction)
        })

    except TypeError:
        return jsonify({'error': 'Invalid input type'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# --- Main Entry Point ---

if __name__ == '__main__':
    # Run the Flask application in debug mode
    # In a production environment, you would not use debug=True
    app.run(debug=True)
