\# AI Text Detection



A machine learning project that detects AI-generated text using two different

classification approaches: Decision Tree (DT) and Multilayer Perceptron ANN (MLP).

Both implementations include a full GUI for easy interaction.



\---



\## Project Structure



&#x20;   AI-text-detection/

&#x20;   |

&#x20;   |-- DT/

&#x20;   |   |-- dt\_final.py             # Model training and evaluation

&#x20;   |   |-- dt\_gui.py               # GUI application

&#x20;   |   |-- extract\_dt.py           # Feature extraction

&#x20;   |   |-- dt\_model.pkl            # Trained DT model

&#x20;   |   |-- dt\_feature\_importance.png

&#x20;   |   |-- dt\_roc\_curve.png

&#x20;   |   └-- results.jsonl

&#x20;   |

&#x20;   |-- MLP/

&#x20;   |   |-- train\_mlp.py            # Model training

&#x20;   |   |-- infer\_mlp.py            # Inference script

&#x20;   |   |-- ann\_gui.py              # GUI application

&#x20;   |   |-- bert\_extract.py         # BERT feature extraction

&#x20;   |   |-- mlp\_model.pkl           # Trained MLP model

&#x20;   |   |-- bert\_pca.pkl            # PCA transformer

&#x20;   |   |-- bert\_scaler.pkl         # Feature scaler

&#x20;   |   |-- mlp\_scaler.pkl          # MLP scaler

&#x20;   |   |-- mlp\_label\_encoder.pkl   # Label encoder

&#x20;   |   └-- mlp\_training\_stats.json

&#x20;   |

&#x20;   |-- final\_report\_AI.pdf

&#x20;   └-- README.md



\---



\## Approaches



\### 1. Decision Tree (DT)

\- Classical ML approach using handcrafted features

\- Feature importance visualization

\- ROC curve evaluation

\- Interactive GUI for real-time detection



\### 2. MLP / ANN

\- Deep learning approach using BERT embeddings

\- PCA dimensionality reduction

\- Trained multilayer perceptron classifier

\- Interactive GUI for real-time detection



\---



\## How to Run



\### Requirements

&#x20;   pip install -r requirements.txt



\### Run Decision Tree GUI

&#x20;   cd DT

&#x20;   python dt\_gui.py



\### Run MLP/ANN GUI

&#x20;   cd MLP

&#x20;   python ann\_gui.py



\---



\## Results



See final\_report\_AI.pdf for full evaluation, metrics, and analysis.



\---



\## Demo



Demo video available on request.



\---



\## Authors



\- Arham Awan 30934

\- Muhammad Ismail 30917

\- Karan Kumar 30212

\- Mustafa Khan 31169



\---



\## License



This project is for academic purposes only.

