import sys
import json
import os
import pandas as pd
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel,
    QLineEdit, QComboBox, QHBoxLayout, QMessageBox, QSpinBox, QCheckBox, QScrollArea
)
from PyQt5.QtCore import Qt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
import numpy as np
import multiprocessing

# Initialize NLTK's VADER
import nltk

# Download NLTK data
nltk.download('vader_lexicon', quiet=True)

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

        # Create a multiprocessing pool
        self.pool = multiprocessing.Pool(initializer=self._worker_init)

        self.initUI()

    @staticmethod
    def _worker_init():
        global sentiment_analyzer
        sentiment_analyzer = SentimentIntensityAnalyzer()

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
        self.results_combo.currentIndexChanged.connect(self.display_conversation)  # Connect signal to slot
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

        # Prepare the data to be passed to the worker function
        data_dict = self.data.to_dict('records')
        num_topics = self.topic_spin.value() if not self.topic_auto.isChecked() else self.auto_select_topics()
        num_emotions = self.emotion_spin.value() if not self.emotion_auto.isChecked() else self.auto_select_emotions()

        # Use the multiprocessing pool for heavy computations
        results = self.pool.apply_async(self._process_data_worker, (data_dict, num_topics, num_emotions))

        # Wait for the results
        processed_data = results.get()

        # Update the class attributes with the processed data
        self.data = pd.DataFrame(processed_data['data'])
        self.semantic_embeddings = processed_data['semantic_embeddings']

        QMessageBox.information(self, "Processing Complete", "Data has been processed and indexed.")

    @staticmethod
    def _process_data_worker(data_dict, num_topics, num_emotions):
        # This method will run in a separate process
        data = pd.DataFrame(data_dict)

        # Topic Modeling
        vectorizer = CountVectorizer(stop_words='english')
        dtm = vectorizer.fit_transform(data['full_text'])
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(dtm)
        topics = lda.transform(dtm)
        topic_labels = [f"Topic {i}" for i in range(num_topics)]
        for i in range(num_topics):
            data[topic_labels[i]] = topics[:, i]

        emotions = ConversationIndexer.analyze_emotions(data['full_text'], sentiment_analyzer, num_emotions)
        for emotion in emotions:
            data[emotion] = emotions[emotion]

        semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        semantic_embeddings = semantic_model.encode(data['full_text'].tolist(), convert_to_numpy=True)

        return {
            'data': data.to_dict('records'),
            'semantic_embeddings': semantic_embeddings
        }

    def auto_select_topics(self):
        # Placeholder for automatic topic selection logic (e.g., coherence scores)
        return 5  # Default value

    def auto_select_emotions(self):
        # Placeholder for automatic emotion category selection
        return 5  # Default value

    @staticmethod
    def analyze_emotions(texts, sentiment_analyzer, num_emotions=5):
        # Simple sentiment analysis using VADER
        emotions = {'positive': [], 'negative': [], 'neutral': []}
        for text in texts:
            scores = sentiment_analyzer.polarity_scores(text)
            emotion = 'positive' if scores['compound'] >= 0.05 else 'negative' if scores['compound'] <= -0.05 else 'neutral'
            for emo in emotions:
                emotions[emo].append(1 if emo == emotion else 0)
        return emotions

    def search(self):
        query = self.search_input.text().strip()
        if not query:
            QMessageBox.warning(self, "Empty Query", "Please enter a search query.")
            return

        # Simple semantic search using cosine similarity
        query_embedding = self.semantic_model.encode([query], convert_to_numpy=True)
        similarities = np.dot(self.semantic_embeddings, query_embedding.T).squeeze()
        sorted_indices = similarities.argsort()[::-1]  # All results, sorted by similarity

        # Clear previous results
        self.clear_results()

        # Create a scroll area for results
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        for idx in sorted_indices:
            convo = self.data.iloc[idx]
            display_text = f"{convo['title']} (Created: {convo['create_time'].strftime('%Y-%m-%d')})"
            result_button = QPushButton(display_text)
            result_button.clicked.connect(lambda _, i=idx: self.display_conversation(i))
            scroll_layout.addWidget(result_button)

        scroll_content.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_content)

        # Add scroll area to the main layout
        self.layout().addWidget(scroll_area)

        QMessageBox.information(self, "Search Complete", "Results have been populated.")

    def clear_results(self):
        # Remove the previous scroll area if it exists
        for i in reversed(range(self.layout().count())):
            widget = self.layout().itemAt(i).widget()
            if isinstance(widget, QScrollArea):
                widget.setParent(None)

    def display_conversation(self, index):
        conversation = self.data.iloc[index]

        # Create a new window to display the conversation
        self.conversation_window = QWidget()
        self.conversation_window.setWindowTitle(conversation['title'])
        self.conversation_window.setGeometry(200, 200, 600, 400)

        layout = QVBoxLayout()

        # Create a scrollable text area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Define colors
        marigold = "#FFA500"  # A warm yellow-orange
        terra_cotta = "#E2725B"  # A reddish-orange
        terra_verde = "#5F8575"  # A muted green
        black = "#000000"

        # Check if 'messages' exists in the conversation
        if 'messages' in conversation:
            for message in conversation['messages']:
                if isinstance(message, dict) and 'text' in message:
                    text = message['text']
                    role = message.get('role', 'Unknown')

                    message_label = QLabel(f"{role.capitalize()}: {text}")
                    message_label.setWordWrap(True)

                    if role.lower() == "assistant":
                        message_label.setStyleSheet(
                            f"background-color: {terra_verde}; color: {marigold}; border-radius: 10px; padding: 10px; margin: 5px;"
                        )
                    elif role.lower() == "user":
                        message_label.setStyleSheet(
                            f"background-color: {terra_cotta}; color: {marigold}; border-radius: 10px; padding: 10px; margin: 5px;"
                        )

                    scroll_content.setStyleSheet(
                        f"background-color: 'navy blue'; color: 'navy blue'; border-radius: 10px; padding: 10px; margin: 5px;"
                    )

                    scroll_layout.addWidget(message_label)
        else:
            error_label = QLabel("No messages found in this conversation.")
            error_label.setStyleSheet(f"color: {terra_cotta};")
            scroll_layout.addWidget(error_label)

        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)

        self.conversation_window.setLayout(layout)
        self.conversation_window.show()

if __name__ == '__main__':
    # Use 'spawn' method for starting processes
    multiprocessing.set_start_method('spawn')

    app = QApplication(sys.argv)
    window = ConversationIndexer()
    window.show()

    # Ensure proper cleanup when the application exits
    try:
        sys.exit(app.exec_())
    finally:
        # Perform any necessary cleanup here
        if hasattr(window, 'pool'):
            window.pool.close()
            window.pool.join()