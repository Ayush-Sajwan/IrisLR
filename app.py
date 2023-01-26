from flask import Flask,request,jsonify

import pickle
import pandas as pd


app = Flask(__name__)

model = pickle.load(open('iris_model.pkl', 'rb'))


@app.route('/iris')
def hello_world():
    sl=request.args.get('sl')
    sw=request.args.get('sw')
    pl=request.args.get('pl')
    pw=request.args.get('pw')
    temp=model.predict(pd.DataFrame([[sl,sw,pl,pw]],columns=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']))
    species="none"


    if temp==0:
        species="Iris-setosa"
    elif temp==1:
        species="Iris-versicolor"
    else:
        species="Iris-virginica"

    obj={ "type":species,"SepalLength":sl,"SepalWidth":sw,"PetalLength":pl,"PetalWidth":pw,"By":"Ayush Sajwan"}

    return jsonify(obj)


if __name__=="__main__":
    app.run(debug=True)