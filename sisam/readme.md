**1. Unified Misinformation Detection Dashboard (Streamlit Frontend)**
This repository contains the interactive frontend dashboard for the social media detection system. This application serves as the primary user interface, connecting to the FastAPI backend to visualize credibility scores for articles and bot probabilities for user profiles.

**2.Key Features**

This dashboard provides a unified, responsive interface for two critical analysis tasks:

*Fake News Classification: The Article Analyzer allows users to paste text and receive a prediction from the DistilBERT model. The dashboard presents the confidence score and detailed class probabilities through clear visualizations.

*Social Bot Detection: The User Analyzer takes inputs for key user metadata (e.g., follower counts, account age). It calculates and displays the bot probability score from the Random Forest model, along with a corresponding risk level (Low, Medium, or High).

*High-Performance Integration: The frontend uses Python's requests library to manage reliable, asynchronous communication with the FastAPI backend, ensuring the application remains fast and responsive.

**3.Dashboard Functionality**

The dashboard is divided into three functional tabs:

*Article Analyzer: This section accepts text input, sends it to the API, and visualizes the model's confidence and class probabilities for credibility assessment.

*User Analyzer: This section collects numerical and boolean features about a profile, sends them for bot scoring, and visualizes the resulting Bot Probability and Risk Level using a progress bar and status badge.

*Diagnostics: This utility tab allows the user to manually test the connection health between the Streamlit frontend and the FastAPI backend.

**4.My Role**

I developed frontend application, including the layout, API integration logic, input forms, and data visualization presentation layer and coordinated with my team for integration with backend. 
