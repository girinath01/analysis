import numpy as np
import pickle
import sys
import types
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle
import time
from flask import Flask, request, jsonify, render_template
import base64
import io

app = Flask(__name__)
model = None
survmodel = None


def register_legacy_pandas_aliases():
    """Provide removed pandas index classes needed by the saved lifelines model."""
    module_name = 'pandas.core.indexes.numeric'
    if module_name not in sys.modules:
        legacy_indexes = types.ModuleType(module_name)
        legacy_indexes.Int64Index = pd.Index
        legacy_indexes.UInt64Index = pd.Index
        legacy_indexes.Float64Index = pd.Index
        sys.modules[module_name] = legacy_indexes


def load_models():
    global model, survmodel
    register_legacy_pandas_aliases()
    if model is None:
        model = pickle.load(open('model.pkl', 'rb'))
    if survmodel is None:
        survmodel = pickle.load(open('survivemodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    try:
        load_models()
    except (ImportError, ModuleNotFoundError, ValueError, AttributeError) as error:
        return render_template(
            'index.html',
            error_text='The prediction models could not be loaded in this environment: {}'.format(error)
        ), 503

    gender = int(request.form.get("gender", 0))
    SeniorCitizen = 0
    if 'SeniorCitizen' in request.form:
        SeniorCitizen = 1
    Partner = 0
    if 'Partner' in request.form:
        Partner = 1
    Dependents = 0
    if 'Dependents' in request.form:
        Dependents = 1
    PaperlessBilling = 0
    if 'PaperlessBilling' in request.form:
        PaperlessBilling = 1

    MonthlyCharges = int(request.form["MonthlyCharges"])
    Tenure = int(request.form["Tenure"])
    TotalCharges = MonthlyCharges*Tenure

    PhoneService = 0
    if 'PhoneService' in request.form:
        PhoneService = 1

    MultipleLines = 0
    if 'MultipleLines' in request.form and PhoneService == 1:
        MultipleLines = 1

    InternetService_Fiberoptic = 0
    InternetService_No = 0
    if int(request.form["InternetService"]) == 0:
        InternetService_No = 1
    elif int(request.form["InternetService"]) == 2:
        InternetService_Fiberoptic = 1

    OnlineSecurity = 0
    if 'OnlineSecurity' in request.form and InternetService_No == 0:
        OnlineSecurity = 1

    OnlineBackup = 0
    if 'OnlineBackup' in request.form and InternetService_No == 0:
        OnlineBackup = 1

    DeviceProtection = 0
    if 'DeviceProtection' in request.form and InternetService_No == 0:
        DeviceProtection = 1

    TechSupport = 0
    if 'TechSupport' in request.form and InternetService_No == 0:
        TechSupport = 1

    StreamingTV = 0
    if 'StreamingTV' in request.form and InternetService_No == 0:
        StreamingTV = 1

    StreamingMovies = 0
    if 'StreamingMovies' in request.form and InternetService_No == 0:
        StreamingMovies = 1

    Contract_Oneyear = 0
    Contract_Twoyear = 0
    if int(request.form["Contract"]) == 1:
        Contract_Oneyear = 1
    elif int(request.form["Contract"]) == 2:
        Contract_Twoyear = 1

    PaymentMethod_CreditCard = 0
    PaymentMethod_ElectronicCheck = 0
    PaymentMethod_MailedCheck = 0
    if int(request.form["PaymentMethod"]) == 1:
        PaymentMethod_CreditCard = 1
    elif int(request.form["PaymentMethod"]) == 2:
        PaymentMethod_ElectronicCheck = 1
    elif int(request.form["PaymentMethod"]) == 3:
        PaymentMethod_MailedCheck = 1

    features = [gender, SeniorCitizen, Partner, Dependents, Tenure, PhoneService, MultipleLines, OnlineSecurity, OnlineBackup,
       DeviceProtection, TechSupport, StreamingTV, StreamingMovies, PaperlessBilling, MonthlyCharges, TotalCharges,
       InternetService_Fiberoptic, InternetService_No, Contract_Oneyear,Contract_Twoyear,
       PaymentMethod_CreditCard, PaymentMethod_ElectronicCheck, PaymentMethod_MailedCheck]

    columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies','PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
       'InternetService_Fiber optic', 'InternetService_No', 'Contract_One year', 'Contract_Two year',
       'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

    final_features = [np.array(features)]
    prediction = model.predict_proba(final_features)

    output = prediction[0,1]

    # Feature importance explanation from the loaded forest.
    explanation_img = io.BytesIO()
    importance = model.feature_importances_
    order = np.argsort(importance)[-8:]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.barh(np.array(columns)[order], importance[order], color='#6558f5')
    ax.set_title('Top factors used by the churn model')
    ax.set_xlabel('Relative importance')
    fig.tight_layout()
    fig.savefig(explanation_img, format='png', bbox_inches='tight')
    plt.close(fig)
    explanation_img.seek(0)
    shap_url = base64.b64encode(explanation_img.getvalue()).decode()

    # Hazard and Survival Analysis
    surv_feats = np.array([gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines, OnlineSecurity, OnlineBackup,
       DeviceProtection, TechSupport, StreamingTV, StreamingMovies, PaperlessBilling, MonthlyCharges, TotalCharges,
       InternetService_Fiberoptic, InternetService_No, Contract_Oneyear,Contract_Twoyear,
       PaymentMethod_CreditCard, PaymentMethod_ElectronicCheck, PaymentMethod_MailedCheck])

    surv_feats = surv_feats.reshape(1, len(surv_feats))

    timeline = np.arange(0, 73)
    try:
        hazard_values = survmodel.predict_cumulative_hazard(surv_feats).iloc[:, 0].to_numpy()
        survival_values = survmodel.predict_survival_function(surv_feats).iloc[:, 0].to_numpy()
        timeline = np.asarray(survmodel.predict_survival_function(surv_feats).index, dtype=float)
        survival_values = np.clip(survival_values, 0, 1)
        hazard_values = np.maximum(hazard_values, 0)
        survival_source = 'Cox survival model'
    except (AttributeError, KeyError, TypeError, ValueError):
        # Old lifelines pickles can load but cannot execute on current lifelines.
        # Keep the app usable with a probability-calibrated survival approximation.
        churn_risk = float(np.clip(output, 0.01, 0.99))
        scale = max(12.0, 72.0 * (1.0 - churn_risk))
        survival_values = np.exp(-timeline / scale)
        hazard_values = timeline / scale
        survival_source = 'Model-compatible approximation'

    hazard_img = io.BytesIO()
    fig, ax = plt.subplots()
    ax.plot(timeline, hazard_values, color='red', label='Cumulative hazard')
    plt.axvline(x=Tenure, color = 'blue', linestyle='--')
    plt.legend(labels=['Hazard', 'Current Position'])
    ax.set_xlabel('Tenure', size = 10)
    ax.set_ylabel('Cumulative Hazard', size = 10)
    ax.set_title('Cumulative Hazard Over Time ({})'.format(survival_source))
    plt.savefig(hazard_img, format = 'png')
    plt.close(fig)
    hazard_img.seek(0)
    hazard_url = base64.b64encode(hazard_img.getvalue()).decode()

    surv_img = io.BytesIO()
    fig, ax = plt.subplots()
    ax.plot(timeline, survival_values, color='red', label='Survival probability')
    plt.axvline(x=Tenure, color = 'blue', linestyle='--')
    plt.legend(labels=['Survival Function','Current Position'])
    ax.set_xlabel('Tenure', size = 10)
    ax.set_ylabel('Survival Probability', size = 10)
    ax.set_title('Survival Probability Over Time ({})'.format(survival_source))
    plt.savefig(surv_img, format = 'png')
    plt.close(fig)
    surv_img.seek(0)
    surv_url = base64.b64encode(surv_img.getvalue()).decode()

    viable_life = timeline[survival_values > 0.1]
    max_life = viable_life.max() if len(viable_life) else 0

    CLTV = max_life * MonthlyCharges

    # Gauge plot
    def degree_range(n):
        start = np.linspace(0,180,n+1, endpoint=True)[0:-1]
        end = np.linspace(0,180,n+1, endpoint=True)[1::]
        mid_points = start + ((end-start)/2.)
        return np.c_[start, end], mid_points

    def rot_text(ang):
        rotation = np.degrees(np.radians(ang) * np.pi / np.pi - np.radians(90))
        return rotation

    def gauge(labels=['LOW','MEDIUM','HIGH','EXTREME'], \
              colors=['#007A00','#0063BF','#FFCC00','#ED1C24'], Probability=1, fname=False):

        N = len(labels)
        colors = colors[::-1]


        """
        begins the plotting
        """

        gauge_img = io.BytesIO()
        fig, ax = plt.subplots()

        ang_range, mid_points = degree_range(4)

        labels = labels[::-1]

        """
        plots the sectors and the arcs
        """
        patches = []
        for ang, c in zip(ang_range, colors):
            # sectors
            patches.append(Wedge((0.,0.), .4, *ang, facecolor='w', lw=2))
            # arcs
            patches.append(Wedge((0.,0.), .4, *ang, width=0.10, facecolor=c, lw=2, alpha=0.5))

        [ax.add_patch(p) for p in patches]


        """
        set the labels (e.g. 'LOW','MEDIUM',...)
        """

        for mid, lab in zip(mid_points, labels):

            ax.text(0.35 * np.cos(np.radians(mid)), 0.35 * np.sin(np.radians(mid)), lab, \
                horizontalalignment='center', verticalalignment='center', fontsize=14, \
                fontweight='bold', rotation = rot_text(mid))

        """
        set the bottom banner and the title
        """
        r = Rectangle((-0.4,-0.1),0.8,0.1, facecolor='w', lw=2)
        ax.add_patch(r)

        ax.text(0, -0.05, 'Churn Probability ' + np.round(Probability,2).astype(str), horizontalalignment='center', \
             verticalalignment='center', fontsize=22, fontweight='bold')

        """
        plots the arrow now
        """

        pos = (1-Probability)*180
        ax.arrow(0, 0, 0.225 * np.cos(np.radians(pos)), 0.225 * np.sin(np.radians(pos)), \
                     width=0.04, head_width=0.09, head_length=0.1, fc='k', ec='k')

        ax.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
        ax.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))

        """
        removes frame and ticks, and makes axis equal and tight
        """

        ax.set_frame_on(False)
        ax.axes.set_xticks([])
        ax.axes.set_yticks([])
        ax.axis('equal')
        plt.tight_layout()

        plt.savefig(gauge_img, format = 'png')
        gauge_img.seek(0)
        url = base64.b64encode(gauge_img.getvalue()).decode()
        return url

    gauge_url = gauge(Probability = output)

    t = time.time()
    return render_template('index.html', prediction_text='Churn probability is {} and Expected Life Time Value is ${}'.format(round(output, 2), CLTV), url_1 = gauge_url, url_2 = shap_url, url_3 = hazard_url, url_4 = surv_url)


if __name__ == "__main__":
    app.run(debug=True)
