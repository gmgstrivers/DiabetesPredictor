import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

app=Flask(__name__)
model=pickle.load(open("model.pkl","rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    float_feature=[float(x) for x in request.form.values()]
    features=[np.array(float_feature)]
    prediction=model.predict(features)

    if int(prediction)==1:
        return render_template("index.html",prediction_text="You Have {}.Please Follow a proper diet.".format("Diabetes"))
    else:
        return render_template("index.html", prediction_text="You don't have {}.Keep your diet well.".format("Diabetes"))


if __name__=="__main__":
    app.run(debug=True)