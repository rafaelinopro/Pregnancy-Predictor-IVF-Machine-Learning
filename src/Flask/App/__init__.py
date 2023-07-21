import pandas as pd
import pickle
from flask import Flask, request, render_template, jsonify

app = Flask(__name__, instance_relative_config=True)
app.config.from_object("config")
app.config.from_pyfile("config.py")


app = Flask(__name__)


@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')


@app.route("/", methods=['POST'])
def procesarsamples():
    nombrearchivo = request.form.get('nombreArchivo')
    if nombrearchivo:
        data = pd.read_csv(f'App/static/data/{nombrearchivo}.csv', index_col=0)
        model = pickle.load(open('App/static/model/my_model.pkl', 'rb'))
        prediccion = model.predict(data)
        proba = model.predict_proba(data)
        proba_cero = proba[:, 0].round(2)
        proba_uno = proba[:, 1].round(2)

        mensaje = f"La probabilidad de embarazo es  {float(proba_uno) * 100:.3f} % (predicción: {int(prediccion)})"

        return jsonify(mensaje=mensaje)

    return render_template('index.html')


@app.route("/input_data", methods=['GET'])
def home_data():
    return render_template('input_data.html')

@app.route("/input_data_PRE", methods=['GET'])
def home_data_pre():
    return render_template('input_data_PRE.html')


@app.route("/pred_results", methods=['POST'])
def input_data():
    cols_name = ['Patient age at treatment',
                 'Total number of previous IVF cycles', 'Total number of previous pregnancies - IVF and DI', 'Total number of previous live births - IVF or DI',
                 'Causes of infertility - tubal disease',
                 'Causes of infertility - ovulatory disorder',
                 'Causes of infertility - male factor',
                 'Causes of infertility - patient unexplained',
                 'Causes of infertility - endometriosis',
                 'Elective single embryo transfer',
                 'Fresh eggs collected',
                 'Total eggs mixed',
                 'Total embryos created',
                 'Embryos transferred',
                 'Total embryos thawed',
                 'Embryos transferred from eggs micro-injected',
                 'Embryos stored for use by patient',
                 'Date of embryo transfer',
                 'Specific treatment type_IVF',
                 'Sperm source_Partner']
    datos_formulario = {}
    if request.form:
        for i, campo in enumerate(request.form):
            if i < len(cols_name):
                datos_formulario[cols_name[i]] = request.form[campo]

        test_data = pd.DataFrame(datos_formulario, index=[0])
        model = pickle.load(open('App/static/model/my_model.pkl', 'rb'))
        prediccion = model.predict(test_data)
        proba = model.predict_proba(test_data)
        proba_cero = proba[:, 0].round(2)
        proba_uno = proba[:, 1].round(2)

        mensaje = f"La probabilidad de embarazo es {float(proba_uno) * 100} % (predicción: {int(prediccion)})"

        return render_template('pred_result.html',mensaje=mensaje)

    return render_template('input_data.html')

@app.route("/pred_result_PRE", methods=['POST'])
def input_data_PRE():
    cols_name = ['Patient age at treatment',
                 'Total number of previous IVF cycles', 'Total number of previous pregnancies - IVF and DI', 'Total number of previous live births - IVF or DI',
                 'Causes of infertility - tubal disease',
                 'Causes of infertility - ovulatory disorder',
                 'Causes of infertility - male factor',
                 'Causes of infertility - patient unexplained',
                 'Causes of infertility - endometriosis',
                 ]
    datos_formulario = {}
    if request.form:
        for i, campo in enumerate(request.form):
            if i < len(cols_name):
                datos_formulario[cols_name[i]] = request.form[campo]

        test_data = pd.DataFrame(datos_formulario, index=[0])
        model = pickle.load(open('App/static/model/my_model_PRE.pkl', 'rb'))
        prediccion = model.predict(test_data)
        proba = model.predict_proba(test_data)
        proba_cero = proba[:, 0].round(2)
        proba_uno = proba[:, 1].round(2)

        mensaje = f"La probabilidad de embarazo es {float(proba_uno) * 100} % (predicción: {int(prediccion)})"
        
        return render_template('pred_result_PRE.html',mensaje=mensaje)

    return render_template('input_data_PRE.html')

@app.route("/ml_explain", methods=['POST', 'GET'])
def explain():
    return render_template('ml_explain.html')

@app.route("/anexos", methods=['POST', 'GET'])
def anexos():
    return render_template('anexos.html')

@app.route("/elements", methods=['POST', 'GET'])
def elements():
    return render_template('elements.html')

@app.route("/predhtml", methods=['POST', 'GET'])
def predhtml():
    return render_template('pred_result_PRE.html')


if __name__ == '__main__':
    app.run(debug=True)
