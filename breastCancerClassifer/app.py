from flask import Flask, request, jsonify,sklearn,gunicorn
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))


app = Flask(__name__)


@app.route('/')
def home():
    return "Hello world"


@app.route('/predict', methods=['POST'])
def predict():
    idNum = request.form.get('ID number')
    RadiusMean = request.form.get('Radius (mean)')
    TextureMean = request.form.get('Texture (mean)')
    PerimeterMean = request.form.get('Perimeter (mean)')
    AreaMean = request.form.get('Area (mean)')
    SmoothnessMean = request.form.get('Smoothness (mean)')
    CompactnessMean = request.form.get('Compactness (mean)')
    ConcavityMean = request.form.get('Concavity (mean)')
    ConcavePointsMean = request.form.get('Concave Points(mean)')
    SymmetryMean = request.form.get('Symmetry (mean)')
    FractalDimensionMean = request.form.get('Fractal Dimension (mean)')
    RadiusStandardError = request.form.get('Radius (standard error)')
    TextureStandardError = request.form.get('Texture (standard error)')
    PerimeterStandardError = request.form.get('Perimeter (standard error)')
    AreaStandardError = request.form.get('Area (standard error')
    SmoothnessStandardError = request.form.get('Smoothness (standard error)')
    CompactnessStandardError = request.form.get('Compactness (standard error)')
    ConcavityStandardError = request.form.get('Concavity (standard error')
    ConcavePointsStandardError = request.form.get('Concave Points (standard error)')
    SymmetryStandardError = request.form.get('Symmetry (standard error)')
    FractalDimensionStandardError = request.form.get('Fractal Dimension (standard error)')
    RadiusWorst = request.form.get('Radius (worst)')
    TextureWorst = request.form.get('Texture (worst)')
    PerimeterWorst = request.form.get('Perimeter (worst')
    AreaWorst = request.form.get('Area (worst)')
    SmoothnessWorst = request.form.get('Smoothness (worst)')
    CompactnessWorst = request.form.get('Compactness (worst)')
    ConcavityWorst = request.form.get('Concavity (worst)')
    ConcavePointsWorst = request.form.get('ConcavePoints (worst)')
    SymmetryWorst = request.form.get('Symmetry (worst)')
    FractalDimensionWorst = request.form.get('Fractal Dimension (worst)')

    result = {'ID number': idNum, 'Radius (mean)': RadiusMean, 'Texture (mean)': TextureMean,
              'Perimeter (mean)': PerimeterMean, 'Area (mean)': AreaMean, 'Smoothness (mean)': SmoothnessMean,
              'Compactness (mean)': CompactnessMean, 'Concavity (mean)': ConcavityMean,
              'Concave Points(mean)': ConcavePointsMean, 'Symmetry (mean)': SymmetryMean,
              'Fractal Dimension (mean)': FractalDimensionMean,
              'Radius (standard error)': RadiusStandardError, 'Texture (standard error)': TextureStandardError,
              'Perimeter (standard error)': PerimeterStandardError, 'Area (standard error)': AreaStandardError,
              'Smoothness (standard error)': SmoothnessStandardError,
              'Compactness (standard error)': CompactnessStandardError,
              'Concavity (standard error)': ConcavityStandardError,
              'Concave Points (standard error)': ConcavePointsStandardError,
              'Symmetry (standard error)': SymmetryStandardError,
              'Fractal Dimension (standard error)': FractalDimensionStandardError, 'Radius (worst)': RadiusWorst,
              'Texture (worst)': TextureWorst, 'Perimeter (worst)': PerimeterWorst, 'Area (worst)': AreaWorst,
              'Smoothness (worst)': SmoothnessWorst,
              'Compactness (worst)': CompactnessWorst, 'Concavity (worst)': ConcavityWorst,
              'ConcavePoints (worst)': ConcavePointsWorst, 'Symmetry (worst)': SymmetryWorst,
              'Fractal Dimension (worst)': FractalDimensionWorst}

    input_query = np.array([[idNum, RadiusMean, TextureMean, PerimeterMean, AreaMean, SmoothnessMean, CompactnessMean,
                             ConcavityMean, ConcavePointsMean, SmoothnessMean, FractalDimensionMean,
                             RadiusStandardError, TextureStandardError, PerimeterStandardError, AreaStandardError,
                             SmoothnessStandardError, CompactnessStandardError, ConcavityStandardError,
                             ConcavePointsStandardError, SymmetryStandardError, FractalDimensionStandardError,
                             RadiusWorst, TextureWorst, PerimeterWorst, AreaWorst, SmoothnessWorst, CompactnessWorst,
                             ConcavityWorst, ConcavePointsWorst, SymmetryWorst, FractalDimensionWorst]])
    result = model.predict(input_query)[1]
    return jsonify({'Diagnose' : str(result)})


