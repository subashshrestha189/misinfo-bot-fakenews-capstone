This folder contains the individually developed code for the Social Bot and Fake News Detection capstone project. These files show my direct contributions to the dashboard user interface, automated testing suite, and API layer, all of which are essential to make the system dependable, useful, and ready for demonstration.

## 1. FastAPI Backend - app.py

The primary FastAPI backend that serves the machine learning models was developed by myself.
Important contributions consist of:
  Building REST endpoints
    -> /analyze/article: uses the trained DistilBERT model to do fake news inference.
    -> /analyze/user: uses Random Forest predictions to compute bot-likelihood.
    -> Preprocessed machine learning models (joblib and safetensors formats) are loaded and managed.

  Creating structured JSON responses with:
    -> Prediction labels
    -> Confidence scores
    -> Important metadata

  For production-ready behavior, error handling, input validation, and response formatting are included.

The main layer of interaction between the ML pipeline and the user interface is represented by this file.

## 2. Streamlit Dashboard - dashboard.py

The interactive dashboard utilized for the project demonstration was created.

Main features:
-> UI easy to use and allows:
  -> Article content submission for fake news analysis
  -> User profile submission for bot detection
-> Real-time responses by calling the FastAPI backend.
-> Article Analyzer, User Analyzer, and Diagnostics are arranged neatly.
-> Model output display including:
  -> Prediction labels
  -> Confidence scores
  -> Error messages (if any)

This dashboard serves as the system's main presentation and demo interface.

## 3. Automated Tests - tests/

I wrote the test scripts that guarantee the dependability of the API:
-> test_api_basic.py
  -> confirms the health of the FastAPI server.
  -> verifies the successful loading of core endpoints.

-> test_api_user.py
  -> sends mock user data for bot prediction.
  -> validates input schema, response structure, model output presence

Throughout the project, the team was able to integrate new code while maintaining stability because of these tests.
