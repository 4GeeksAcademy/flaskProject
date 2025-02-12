from flask import Flask, request, render_template
from pickle import load
import pandas as pd

app = Flask(__name__)

# Load your new model (update the path as needed)
model = load(open("/workspaces/flaskProject/src/model_2.pkl", "rb"))

# Map predicted numeric output to a human-readable label.
# Adjust the dictionary to match your model's target encoding.
class_dict = {
    "1": "Wine Class 1",
    "2": "Wine Class 2",
    "3": "Wine Class 3"
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Extract each numeric input from the form.
            # Ensure that the names here match the 'name' attributes in your HTML form.
            alcohol = float(request.form["alcohol"])
            malic_acid = float(request.form["malic_acid"])
            ash = float(request.form["ash"])
            alcalinity_of_ash = float(request.form["alcalinity_of_ash"])
            magnesium = float(request.form["magnesium"])
            total_phenols = float(request.form["total_phenols"])
            flavanoids = float(request.form["flavanoids"])
            nonflavanoid_phenols = float(request.form["nonflavanoid_phenols"])
            proanthocyanins = float(request.form["proanthocyanins"])
            color_intensity = float(request.form["color_intensity"])
            hue = float(request.form["hue"])
            od280_od315 = float(request.form["od280/od315_of_diluted_wines"])
            proline = float(request.form["proline"])
            
            # Create a dictionary for the input data
            data = {
                "alcohol": [alcohol],
                "malic_acid": [malic_acid],
                "ash": [ash],
                "alcalinity_of_ash": [alcalinity_of_ash],
                "magnesium": [magnesium],
                "total_phenols": [total_phenols],
                "flavanoids": [flavanoids],
                "nonflavanoid_phenols": [nonflavanoid_phenols],
                "proanthocyanins": [proanthocyanins],
                "color_intensity": [color_intensity],
                "hue": [hue],
                "od280/od315_of_diluted_wines": [od280_od315],
                "proline": [proline]
            }
            
            # Convert the dictionary to a pandas DataFrame
            input_df = pd.DataFrame(data)
            
            # Make a prediction using your model
            prediction = model.predict(input_df)[0]
            # Convert the prediction to string in case it's numeric
            prediction_str = str(prediction)
            pred_class = class_dict.get(prediction_str, "Unknown")
        except Exception as e:
            print("Error processing form data:", e)
            pred_class = "Error in input data. Please check your inputs."
    else:
        pred_class = None

    return render_template("index.html", prediction=pred_class)

if __name__ == "__main__":
    app.run(debug=True)
