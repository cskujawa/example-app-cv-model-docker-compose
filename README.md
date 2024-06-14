Used this example project [here](https://github.com/streamlit/example-app-cv-model) as a foundation for this app.
I forked that repo to made it into a docker compose project, added in additional CV models to compare, set the results to sort by the confidence score, and if 3 or more models had 50% or greater confidence it gets a poem written by ChatGPT.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/streamlit/example-app-cv-model/main)

# 📷 Computer vision app

This repository contains the source code for a computer vision app built using Streamlit! The app enables a user to upload images (from file, webcam or URL) and see how some famous pre-trained deep learning models would classify them!

<img width="600" alt="image" src="https://github.com/cskujawa/example-app-cv-model-docker-compose/blob/main/CV-App.png">

### Prerequisites

Before you start, ensure you have the following installed:
- Docker (for containerization)

### Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/cskujawa/example-app-cv-model-docker-compose.git
cd example-app-cv-model-docker-compose
```
### Running the Application
For Docker users, build the Docker image and run the container:

```bash
docker compose up -d
```
