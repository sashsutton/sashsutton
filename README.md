# Sasha Sutton

I'm starting the MSc **Data, Knowledge and Hybrid AI (DKAI)** at Université Paris-Saclay this September. Before that: a BSc in Mathematics & Computer Science at Aix-Marseille University, and before *that*, a music production degree in London, where I spent two years on the sound engineering team at Fabric. Not the usual route into AI. I originally went back to study maths just to understand how computers actually work, and ended up staying for the whole thing.

I've just finished a research internship at LIS (Laboratoire d'Informatique et Systèmes, Luminy), working on theme extraction from web archives: pulling WARC files of [Marsactu](https://marsactu.fr) out of the French National Library's archive and tracking how its topics evolved over time with LDA, BERTopic and CamemBERT.

What I want to do long term is research in NLP and computer vision. The music hasn't gone anywhere either — I still produce electronic music and play guitar, and it keeps leaking into my projects.

**Elsewhere:** [portfolio](https://sashasutton.dev/en) · [LinkedIn](https://www.linkedin.com/in/sashasutton4) · [email](mailto:sashsuttons3@icloud.com)

## Projects

### [WARC NLP Pipeline](https://github.com/sashsutton/warc-nlp-pipeline)
The code from my LIS internship. Extracts and cleans web archive captures (Trafilatura, SolR Wayback) into DataFrames, then runs topic modelling to track how the themes of a French investigative outlet shifted over the years. Python, Pandas, BERTopic, CamemBERT, LDA.

### [ML Library](https://github.com/sashsutton/ML_library)
A small neural network library written in C with no dependencies: matrix ops, ReLU/Sigmoid, dense layers, MSE. I wrote a paper on backpropagation (below) and wanted to check I could actually implement the thing I was describing.

### [Air Harp](https://github.com/sashsutton/computer_vision_air_harp)
Point your index finger at a webcam and play a harp in the air. MediaPipe hand tracking + OpenCV, sound with Pygame. The most direct collision of my two backgrounds so far.

### [Vector Search Engine](https://github.com/sashsutton/vector-search-engine)
Semantic search over documents using cosine similarity on sentence-transformer embeddings, served with FastAPI and split into small services on Render/Vercel.

### [Koda Marketplace](https://github.com/koda-lab/koda-web-app) — live at [kodas.works](https://www.kodas.works/en)
A marketplace for no-code automations, built with a team (Next.js, TypeScript, MongoDB). My part was the financial layer: Stripe Connect with an 85/15 commission split, refund reversals, real-time messaging with Pusher, and file delivery through signed S3 URLs.

## Paper

**De la propagation avant à la rétropropagation : une analyse des mécanismes d'apprentissage dans les réseaux de neurones multicouches** — S. Sutton, hal.science, 2026.
A walk through the mathematics of learning in multilayer networks, from the Rosenblatt perceptron to backpropagation. ([Google Scholar](https://scholar.google.com/scholar?q=Sasha+Sutton))

## Tools I use most

Python (NumPy, Pandas, PyTorch), C, TypeScript/Next.js, Java, SQL. FastAPI for serving things, AWS/MongoDB when a project needs a backend.
