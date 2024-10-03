import sys
import json
import os
import pandas as pd
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel,
    QLineEdit, QComboBox, QHBoxLayout, QMessageBox, QSpinBox, QCheckBox, QScrollArea
)
from PyQt5.QtCore import Qt, QThreadPool, QRunnable, pyqtSignal, QObject
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
import numpy as np
import hashlib
import logging
import pickle

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize NLTK's VADER
import nltk

# Download NLTK data
nltk.download('vader_lexicon', quiet=True)

# Add this new class for our worker
class Worker(QRunnable):
    class Signals(QObject):
        finished = pyqtSignal(object)
        error = pyqtSignal(Exception)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = Worker.Signals()

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except Exception as e:
            self.signals.error.emit(e)

class ConversationIndexer(QWidget):
    def __init__(self):
        super().__init__()
        logger.info("Initializing ConversationIndexer")
        self.setWindowTitle("Conversation Indexer")
        self.setGeometry(100, 100, 800, 600)
        self.data = pd.DataFrame()
        self.lda_model = None
        self.vectorizer = None
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.semantic_embeddings = {}  # Change to a dictionary to store embeddings per file
        self.use_cache = True

        self.loaded_files = {}  # Change this to a dictionary

        self.cache_dir = self.get_cache_dir()

        self.threadpool = QThreadPool()
        logger.info(f"Multithreading with maximum {self.threadpool.maxThreadCount()} threads")

        self.initUI()

    def get_cache_dir(self):
        logger.debug("Getting cache directory")
        # Check for a custom cache directory in a config file or environment variable
        custom_cache_dir = os.environ.get('CHAT_SEARCH_CACHE_DIR')
        if custom_cache_dir:
            cache_dir = custom_cache_dir
        else:
            cache_dir = os.path.join(os.path.expanduser("~"), ".chat_search_cache")
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Using cache directory: {cache_dir}")
        return cache_dir

    def closeEvent(self, event):
        logger.info("Close event triggered")
        event.accept()

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

        # Results Area
        self.results_area = QScrollArea()
        self.results_area.setWidgetResizable(True)
        self.results_content = QWidget()
        self.results_layout = QVBoxLayout(self.results_content)
        self.results_area.setWidget(self.results_content)
        layout.addWidget(self.results_area)

        # Cache Usage Checkbox
        self.cache_checkbox = QCheckBox("Use Cache")
        self.cache_checkbox.setChecked(True)
        self.cache_checkbox.stateChanged.connect(self.toggle_cache_usage)
        layout.addWidget(self.cache_checkbox)

        self.setLayout(layout)

    def toggle_topic_spin(self, state):
        self.topic_spin.setEnabled(not state)

    def toggle_emotion_spin(self, state):
        self.emotion_spin.setEnabled(not state)

    def toggle_cache_usage(self, state):
        self.use_cache = state == Qt.Checked

    def load_json(self):
        logger.info("Loading JSON files")
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(self, "Load JSON Files", "",
                                                "JSON Files (*.json);;All Files (*)", options=options)
        if files:
            all_data = []
            for file in files:
                logger.debug(f"Processing file: {file}")
                file_hash = self.calculate_file_hash(file)
                if file in self.loaded_files and self.loaded_files[file] == file_hash:
                    logger.info(f"File {file} already loaded and unchanged. Skipping.")
                    continue
                try:
                    file_data = self.load_or_process_file(file)
                    all_data.extend(file_data)
                    self.loaded_files[file] = file_hash
                    logger.info(f"Successfully processed {len(file_data)} conversations from {file}")
                except Exception as e:
                    logger.error(f"Error processing file {file}: {str(e)}", exc_info=True)
                    QMessageBox.warning(self, "Error", f"Failed to process {file}: {str(e)}")

            if all_data:
                self.data = pd.DataFrame(all_data)
                logger.debug("Sorting data by create_time")
                self.data.sort_values('create_time', ascending=False, inplace=True)
                logger.debug("Dropping duplicates based on conversation_id")
                original_len = len(self.data)
                self.data.drop_duplicates(subset=['conversation_id'], keep='first', inplace=True)
                logger.info(f"Removed {original_len - len(self.data)} duplicate conversations")
                QMessageBox.information(self, "Success", f"Loaded {len(self.data)} unique conversations from {len(files)} files.")
                self.display_all_conversations()
            else:
                logger.warning("No valid conversations were loaded")
                QMessageBox.warning(self, "No Data", "No valid conversations were loaded.")

    def load_or_process_file(self, file):
        logger.debug(f"Loading or processing file: {file}")
        file_content_hash = self.calculate_file_hash(file)
        stable_id = self.get_stable_file_id(file)
        file_mod_time = os.path.getmtime(file)  # Get the file's modification time
        cache_key = f"{file_content_hash}_{stable_id}_{file_mod_time}"
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.chatlog_cache")

        file_data = []

        if self.use_cache and os.path.exists(cache_file):
            logger.debug(f"Attempting to load from cache: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                if self.validate_cache(cached_data, file, file_content_hash):
                    file_data = cached_data['data']
                    if 'semantic_embeddings' in cached_data:
                        self.semantic_embeddings = cached_data['semantic_embeddings']
                    logger.info(f"Loaded {len(file_data)} conversations and semantic embeddings from cache for {file}")
                else:
                    logger.warning(f"Cache file {cache_file} is invalid or outdated")
            except Exception as e:
                logger.error(f"Error loading cache file {cache_file}: {str(e)}", exc_info=True)

        if not file_data:
            logger.info(f"Processing JSON file: {file}")
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                file_data = self.process_json_data(json_data, file)
                logger.info(f"Processed {len(file_data)} conversations from {file}")

                if self.use_cache:
                    logger.debug(f"Updating cache file: {cache_file}")
                    cache_content = {
                        'data': file_data,
                        'file_hash': file_content_hash,
                        'stable_id': stable_id,
                        'timestamp': datetime.now().isoformat(),
                        'semantic_embeddings': self.semantic_embeddings.get(file)
                    }
                    with open(cache_file, 'wb') as f:
                        pickle.dump(cache_content, f)
                    logger.info(f"Updated cache file {cache_file} with {len(file_data)} conversations and semantic embeddings")
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error in file {file}: {str(e)}", exc_info=True)
                raise ValueError(f"Invalid JSON in file {file}: {str(e)}")
            except Exception as e:
                logger.error(f"Error processing file {file}: {str(e)}", exc_info=True)
                raise

        return file_data

    def get_stable_file_id(self, file_path):
        """Get a stable identifier for the file."""
        try:
            # On Unix-like systems, use inode number
            return os.stat(file_path).st_ino
        except AttributeError:
            # On Windows, use file index (similar to inode)
            file_index = os.stat(file_path).st_file_attributes
            return f"{file_index}"

    def validate_cache(self, cached_data, source_file, current_file_hash):
        """Validate the cache content."""
        if not isinstance(cached_data, dict):
            return False
        required_keys = {'file_hash', 'data', 'timestamp', 'stable_id'}
        if not all(key in cached_data for key in required_keys):
            return False
        if cached_data['file_hash'] != current_file_hash:
            return False
        if cached_data['stable_id'] != self.get_stable_file_id(source_file):
            return False
        if not isinstance(cached_data['data'], list) or not cached_data['data']:
            return False
        required_data_keys = {"type", "audio_modality_interaction", "image_modality_interaction",
                              "title", "create_time", "messages", "full_text", "source_file", "conversation_id"}
        return all(isinstance(item, dict) and required_data_keys.issubset(item.keys()) for item in cached_data['data'])

    def calculate_file_hash(self, file_path):
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def process_json_data(self, json_data, source_file):
        processed_data = []
        for convo in json_data:
            record = {
                "type": convo.get("type", ""),
                "audio_modality_interaction": convo.get("audio_modality_interaction", False),
                "image_modality_interaction": convo.get("image_modality_interaction", False),
                "title": convo.get("title", ""),
                "create_time": datetime.fromtimestamp(convo.get("create_time", 0)),
                "messages": convo.get("messages", []),
                "source_file": source_file,
                "conversation_id": self.generate_conversation_id(convo)
            }
            texts = " ".join([msg.get("text", "") for msg in record["messages"]])
            record["full_text"] = texts
            processed_data.append(record)
        return processed_data

    def generate_conversation_id(self, convo):
        # Create a unique ID based only on conversation content
        convo_content = json.dumps(convo, sort_keys=True).encode('utf-8')
        return hashlib.md5(convo_content).hexdigest()

    def process_data(self):
        logger.info("Starting data processing")
        if self.data.empty:
            logger.warning("No data to process")
            QMessageBox.warning(self, "No Data", "Please load JSON files first.")
            return

        data_dict = self.data.to_dict('records')
        num_topics = self.topic_spin.value() if not self.topic_auto.isChecked() else self.auto_select_topics()
        num_emotions = self.emotion_spin.value() if not self.emotion_auto.isChecked() else self.auto_select_emotions()

        logger.debug(f"Processing data with {num_topics} topics and {num_emotions} emotions")

        worker = Worker(self._process_data_worker, data_dict, num_topics, num_emotions, list(self.semantic_embeddings.values()))
        worker.signals.finished.connect(self.process_complete)
        worker.signals.error.connect(self.process_error)

        self.threadpool.start(worker)

    def process_complete(self, result):
        logger.info("Data processing completed successfully")
        self.data = pd.DataFrame(result['data'])
        combined_embeddings = result['semantic_embeddings']

        # Update the cache with the new semantic embeddings
        if self.use_cache:
            start_idx = 0
            for file, file_hash in self.loaded_files.items():
                stable_id = self.get_stable_file_id(file)
                file_mod_time = os.path.getmtime(file)
                cache_key = f"{file_hash}_{stable_id}_{file_mod_time}"
                cache_file = os.path.join(self.cache_dir, f"{cache_key}.chatlog_cache")

                if os.path.exists(cache_file):
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                    file_data_length = len(cached_data['data'])
                    cached_data['semantic_embeddings'] = combined_embeddings[start_idx:start_idx + file_data_length]
                    with open(cache_file, 'wb') as f:
                        pickle.dump(cached_data, f)
                    logger.info(f"Updated cache file {cache_file} with new semantic embeddings")
                    start_idx += file_data_length

        self.semantic_embeddings = combined_embeddings  # This is now a numpy array, not a dict
        QMessageBox.information(self, "Processing Complete", "Data has been processed and indexed.")
        self.display_all_conversations()

    def process_error(self, error):
        logger.error(f"Error during processing: {str(error)}", exc_info=True)
        QMessageBox.critical(self, "Error", f"An error occurred during processing: {str(error)}")

    @staticmethod
    def _process_data_worker(data_dict, num_topics, num_emotions, existing_embeddings):
        logger.info(f"Worker process started with {num_topics} topics and {num_emotions} emotions")
        data = pd.DataFrame(data_dict)

        try:
            logger.debug("Performing topic modeling")
            vectorizer = CountVectorizer(stop_words='english')
            dtm = vectorizer.fit_transform(data['full_text'])
            lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
            lda.fit(dtm)
            topics = lda.transform(dtm)
            topic_labels = [f"Topic {i}" for i in range(num_topics)]
            for i in range(num_topics):
                data[topic_labels[i]] = topics[:, i]

            logger.debug("Analyzing emotions")
            sentiment_analyzer = SentimentIntensityAnalyzer()
            emotions = ConversationIndexer.analyze_emotions(data['full_text'], sentiment_analyzer, num_emotions)
            for emotion in emotions:
                data[emotion] = emotions[emotion]

            logger.debug("Generating semantic embeddings")
            semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            new_embeddings = semantic_model.encode(data['full_text'].tolist(), convert_to_numpy=True)

            # Combine new embeddings with existing ones
            if existing_embeddings:
                semantic_embeddings = np.vstack([*existing_embeddings, new_embeddings])
            else:
                semantic_embeddings = new_embeddings

            logger.info("Worker process completed successfully")
            return {
                'data': data.to_dict('records'),
                'semantic_embeddings': semantic_embeddings
            }
        except Exception as e:
            logger.error(f"Error in worker process: {str(e)}", exc_info=True)
            raise

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
            self.display_all_conversations()
            return

        if self.semantic_embeddings is None or not isinstance(self.semantic_embeddings, np.ndarray):
            logger.warning("Semantic embeddings not available or in incorrect format. Please process the data first.")
            QMessageBox.warning(self, "Search Error", "Please process the data before searching.")
            return

        # Simple semantic search using cosine similarity
        query_embedding = self.semantic_model.encode([query], convert_to_numpy=True)
        similarities = np.dot(self.semantic_embeddings, query_embedding.T).squeeze()
        sorted_indices = similarities.argsort()[::-1]  # All results, sorted by similarity

        self.display_filtered_conversations(sorted_indices)

    def display_all_conversations(self):
        self.clear_results()
        for idx in range(len(self.data)):
            self.add_conversation_button(idx)

    def display_filtered_conversations(self, indices):
        self.clear_results()
        for idx in indices:
            self.add_conversation_button(idx)

    def clear_results(self):
        for i in reversed(range(self.results_layout.count())):
            widget = self.results_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

    def add_conversation_button(self, idx):
        convo = self.data.iloc[idx]
        display_text = f"{convo['title']} (Created: {convo['create_time'].strftime('%Y-%m-%d')})"
        result_button = QPushButton(display_text)
        result_button.clicked.connect(lambda _, i=idx: self.display_conversation(i))
        self.results_layout.addWidget(result_button)

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
    try:
        app = QApplication(sys.argv)
        window = ConversationIndexer()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        logger.critical(f"Critical error in main application: {str(e)}", exc_info=True)
        QMessageBox.critical(None, "Critical Error", f"A critical error occurred: {str(e)}\nPlease check the log file for details.")