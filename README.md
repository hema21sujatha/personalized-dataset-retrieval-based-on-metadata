# personalized-dataset-retrieval-based-on-metadata- Smart, adaptive, and feedback-driven dataset search system
📚 Personalized AI/ML Dataset Retrieval System
This project implements a personalized AI/ML dataset retrieval system that ranks datasets based on user preferences across multiple metadata factors such as Size, Citations, Diversity, Recency, and Semantic Similarity.

Unlike traditional platforms like Kaggle or Hugging Face, this system uses advanced ranking models (TOPSIS), semantic embeddings (SentenceTransformer), and interactive user feedback to deliver highly relevant dataset recommendations.

🚀 Features
Semantic Search: Enter natural language queries to search datasets (e.g., "image classification", "speech processing").

User-Controlled Personalization: Adjust importance weights for Size, Citations, Diversity, Recency, and Similarity.

Metadata Quality Scoring: Penalizes incomplete or unreliable datasets to boost high-quality options.

Dynamic Ranking with TOPSIS: Multi-metadata ranking model using ideal best/worst analysis.

Interactive Feedback Loop: Like datasets and view live ranking evaluation with NDCG@10 and MRR metrics.

Lightweight Deployment: Fully functional via Streamlit Cloud or local deployment.

📊 Future Enhancements
Add content-based image/audio search using CLIP/VIT embeddings.

Introduce reinforcement learning to dynamically tune user preferences.

Implement Pareto front optimization for multi-objective dataset discovery.

✨ Acknowledgments
Sentence Transformers - for embedding-based semantic similarity.

Streamlit - for fast and elegant app development.

TOPSIS - as a multi-criteria decision-making tool for ranking datasets.

OpenML, Hugging Face, Kaggle - sources for dataset metadata.

✅ Quick Note
If you like this project, please ⭐️ star the repository on GitHub!
![interface](https://github.com/user-attachments/assets/9034a6e3-4415-49e6-a30e-00eb90e25c3c)

