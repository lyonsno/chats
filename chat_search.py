import sys
import json
import os
import pandas as pd
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel,
    QLineEdit, QComboBox, QHBoxLayout, QMessageBox, QSpinBox, QCheckBox
)
from PyQt5.QtCore import Qt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize NLTK's VADER
import nltk
nltk.download('vader_lexicon')

class ConversationIndexer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Conversation Indexer")
        self.setGeometry(100, 100, 800, 600)
        self.data = pd.DataFrame()
        self.lda_model = None
        self.vectorizer = None
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.semantic_embeddings = None

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Load JSON Button
        self.load_button = QPushButton("Load JSON Files")
        self.load_button.clicked.connect(self.load_json)
        layout.addWidget(self.load_button)

        # Vocabulary Size for Topics
        topic_layout = QHBoxLayout()
        topic_label = QLabel("Number of Topics:")
        self.topic_spin = QSpinBox()
        self.topic_spin.setMinimum(1)
        self.topic_spin.setMaximum(100)
        self.topic_spin.setValue(5)
        self.topic_auto = QCheckBox("Auto-select")
        self.topic_auto.stateChanged.connect(self.toggle_topic_spin)
        topic_layout.addWidget(topic_label)
        topic_layout.addWidget(self.topic_spin)
        topic_layout.addWidget(self.topic_auto)
        layout.addLayout(topic_layout)

        # Vocabulary Size for Emotions
        emotion_layout = QHBoxLayout()
        emotion_label = QLabel("Number of Emotion Categories:")
        self.emotion_spin = QSpinBox()
        self.emotion_spin.setMinimum(1)
        self.emotion_spin.setMaximum(20)
        self.emotion_spin.setValue(5)
        self.emotion_auto = QCheckBox("Auto-select")
        self.emotion_auto.stateChanged.connect(self.toggle_emotion_spin)
        emotion_layout.addWidget(emotion_label)
        emotion_layout.addWidget(self.emotion_spin)
        emotion_layout.addWidget(self.emotion_auto)
        layout.addLayout(emotion_layout)

        # Process Button
        self.process_button = QPushButton("Process and Index")
        self.process_button.clicked.connect(self.process_data)
        layout.addWidget(self.process_button)

        # Filters
        filter_layout = QHBoxLayout()

        # Creation Date Filter
        self.creation_date_input = QLineEdit()
        self.creation_date_input.setPlaceholderText("Creation Date (YYYY-MM-DD)")
        filter_layout.addWidget(QLabel("Creation Date:"))
        filter_layout.addWidget(self.creation_date_input)

        # Update Date Filter
        self.update_date_input = QLineEdit()
        self.update_date_input.setPlaceholderText("Update Date (YYYY-MM-DD)")
        filter_layout.addWidget(QLabel("Update Date:"))
        filter_layout.addWidget(self.update_date_input)

        # Modality Filter
        self.modality_combo = QComboBox()
        self.modality_combo.addItems(["All", "Audio", "Image"])
        filter_layout.addWidget(QLabel("Modality:"))
        filter_layout.addWidget(self.modality_combo)

        layout.addLayout(filter_layout)

        # Search Bar
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter search query...")
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.search)
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.search_button)
        layout.addLayout(search_layout)

        # Search Results
        self.results_combo = QComboBox()
        layout.addWidget(QLabel("Search Results:"))
        layout.addWidget(self.results_combo)

        self.setLayout(layout)

    def toggle_topic_spin(self, state):
        self.topic_spin.setEnabled(not state)

    def toggle_emotion_spin(self, state):
        self.emotion_spin.setEnabled(not state)

    def load_json(self):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(self, "Load JSON Files", "",
                                                "JSON Files (*.json);;All Files (*)", options=options)
        if files:
            all_data = []
            for file in files:
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                        for convo in json_data:
                            record = {
                                "type": convo.get("type", ""),
                                "audio_modality_interaction": convo.get("audio_modality_interaction", False),
                                "image_modality_interaction": convo.get("image_modality_interaction", False),
                                "title": convo.get("title", ""),
                                "create_time": datetime.fromtimestamp(convo.get("create_time", 0)),
                                "messages": convo.get("messages", [])
                            }
                            # Concatenate all message texts
                            texts = " ".join([msg.get("text", "") for msg in record["messages"]])
                            record["full_text"] = texts
                            all_data.append(record)
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to load {file}: {str(e)}")
            self.data = pd.DataFrame(all_data)
            QMessageBox.information(self, "Success", f"Loaded {len(self.data)} conversations.")

    def process_data(self):
        if self.data.empty:
            QMessageBox.warning(self, "No Data", "Please load JSON files first.")
            return

        # Topic Modeling
        num_topics = self.topic_spin.value() if not self.topic_auto.isChecked() else self.auto_select_topics()
        vectorizer = CountVectorizer(stop_words='english')
        dtm = vectorizer.fit_transform(self.data['full_text'])
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(dtm)
        topics = lda.transform(dtm)
        topic_labels = [f"Topic {i}" for i in range(num_topics)]
        for i in range(num_topics):
            self.data[topic_labels[i]] = topics[:, i]

        # Emotion Analysis
        emotions = self.analyze_emotions(self.data['full_text'], num_emotions=self.emotion_spin.value() if not self.emotion_auto.isChecked() else self.auto_select_emotions())
        for emotion in emotions:
            self.data[emotion] = emotions[emotion]

        # Semantic Embeddings
        self.semantic_embeddings = self.semantic_model.encode(self.data['full_text'].tolist(), convert_to_numpy=True)

        QMessageBox.information(self, "Processing Complete", "Data has been processed and indexed.")

    def auto_select_topics(self):
        # Placeholder for automatic topic selection logic (e.g., coherence scores)
        return 5  # Default value

    def auto_select_emotions(self):
        # Placeholder for automatic emotion category selection
        return 5  # Default value

    def analyze_emotions(self, texts, num_emotions=5):
        # Simple sentiment analysis using VADER
        # For multiple emotions, you might need a more sophisticated model
        emotions = {'positive': 0, 'negative': 0, 'neutral': 0}
        emotion_data = {'positive': [], 'negative': [], 'neutral': []}

        for text in texts:
            scores = self.sentiment_analyzer.polarity_scores(text)
            if scores['compound'] >= 0.05:
                emotion = 'positive'
            elif scores['compound'] <= -0.05:
                emotion = 'negative'
            else:
                emotion = 'neutral'
            emotion_data[emotion].append(1)
            for emo in emotions:
                if emo != emotion:
                    emotion_data[emo].append(0)

        return emotion_data

    def search(self):
        query = self.search_input.text().strip()
        if not query:
            QMessageBox.warning(self, "Empty Query", "Please enter a search query.")
            return

        # Simple semantic search using cosine similarity
        query_embedding = self.semantic_model.encode([query], convert_to_numpy=True)
        similarities = np.dot(self.semantic_embeddings, query_embedding.T).squeeze()
        top_indices = similarities.argsort()[-5:][::-1]  # Top 5 results

        self.results_combo.clear()
        for idx in top_indices:
            convo = self.data.iloc[idx]
            display_text = f"{convo['title']} (Created: {convo['create_time'].strftime('%Y-%m-%d')})"
            self.results_combo.addItem(display_text)

        QMessageBox.information(self, "Search Complete", "Top results have been populated.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ConversationIndexer()
    window.show()
    sys.exit(app.exec_())