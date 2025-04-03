from flask import Flask, request, render_template
from src.pipelines.predict_pipeline import PredictPipeline
from src.configuration.predict_config import CustomData
from src.configuration import config

app = Flask(__name__, template_folder=config.TEMPLATES_DIR)


@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    """Handle prediction requests via form submission."""
    if request.method == 'GET':
        return render_template('home.html')

    try:
        # Extract and validate input data
        data = CustomData(
            gender=request.form.get('gender', '').strip(),
            race_ethnicity=request.form.get('ethnicity', '').strip(),
            parental_level_of_education=request.form.get('parental_level_of_education', '').strip(),
            lunch=request.form.get('lunch', '').strip(),
            test_preparation_course=request.form.get('test_preparation_course', '').strip(),
            reading_score=int(request.form.get('reading_score', 0)),  # Fixed incorrect field mapping
            writing_score=int(request.form.get('writing_score', 0))   # Fixed incorrect field mapping
        )

        # Convert input data to DataFrame
        pred_df = data.get_data_as_dataframe()

        # Debugging log
        print(f"Received Data: {pred_df.to_dict(orient='records')}")

        # Model Prediction
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template('home.html', results=results[0])

    except Exception as e:
        print(f"Error: {e}")  # Replace with proper logging in production
        return render_template('home.html', error="Invalid input or prediction error.")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
