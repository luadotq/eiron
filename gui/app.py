import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLineEdit, QTextEdit, QLabel, QFileDialog, QCheckBox, QListWidget, QListWidgetItem,
    QProgressBar, QSplitter, QTabWidget, QSpinBox, QDoubleSpinBox, QComboBox,
    QGroupBox, QFormLayout, QMessageBox, QMainWindow
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from core.indexer import extract_documents, Indexer
from core.optimized_indexer import OptimizedIndexer
from core.search import Searcher
from semantic.search import EnhancedSemanticSearcher, SemanticSearchConfig
from core.memory_manager import MemoryManager
from core.optimized_memory_manager import OptimizedMemoryManager
from core.system_monitor import SystemMonitor
import logging
from datetime import datetime
import json
from pathlib import Path
import time
import os

class LogHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        self._buffer = []
        self._timer = QTimer()
        self._timer.timeout.connect(self._flush_buffer)
        self._timer.start(100)

    def _flush_buffer(self):
        if self._buffer:
            self.text_widget.append('\n'.join(self._buffer))
            self._buffer.clear()

    def emit(self, record):
        msg = self.format(record)
        self._buffer.append(msg)

class IndexThread(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    log = pyqtSignal(str)

    def __init__(self, folder, indexer_type='standard', index_file='eiron_index.bin', memory_manager=None):
        super().__init__()
        self.folder = folder
        self.indexer_type = indexer_type
        self.index_file = index_file
        self.memory_manager = memory_manager or OptimizedMemoryManager()
        

        if indexer_type == 'optimized':
            self.indexer = OptimizedIndexer(index_file=index_file)
            self.log.emit(f"Using optimized indexer: {index_file}")
        else:
            self.indexer = Indexer(index_file=index_file)
            self.log.emit(f"Using standard indexer: {index_file}")
            

        if not self.memory_manager:
            self.memory_manager = OptimizedMemoryManager()

    def run(self):
        try:
            self.log.emit(f"Starting indexing of directory: {self.folder}")
            self.log.emit(f"Indexer type: {self.indexer_type}")
            self.log.emit(f"Memory manager: OptimizedMemoryManager")
            
            count = self.indexer.index_directory(self.folder)
            self.log.emit(f"Indexing completed. Indexed {count} files.")
            

            self.indexer.save(self.index_file)
            self.log.emit(f"Index saved to: {self.index_file}")
            
            self.finished.emit()
        except Exception as e:
            self.log.emit(f"Error during indexing: {str(e)}")
            self.error.emit(str(e))

class SearchThread(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    result = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, query, use_semantic=False, index_file=None, indexer_type='standard'):
        super().__init__()
        self.query = query
        self.use_semantic = use_semantic
        self.index_file = index_file
        self.indexer_type = indexer_type
        

        if indexer_type == 'optimized':
            self.indexer = OptimizedIndexer(index_file=index_file or 'eiron_index.bin')
        else:
            self.indexer = Indexer(index_file=index_file or 'eiron_index.bin')
            
        if index_file and os.path.exists(index_file):
            self.indexer.load(index_file)

    def run(self):
        try:

            results = self.indexer.search(
                self.query,
                use_semantic=self.use_semantic,
                min_similarity=0.1
            )
            
            if not results:
                self.error.emit("No results found")
                return
            

            for path, score, context in results:
                result_text = (
                    f"File: {path}\n"
                    f"Relevance: {score:.2f}\n"
                )
                if context:
                    result_text += f"Context: {context}\n"
                result_text += f"{'='*50}\n"
                self.result.emit(result_text)
            
            self.finished.emit()
        except Exception as e:
            logging.error(f"Search error: {str(e)}")
            self.error.emit(str(e))

class SettingsTab(QWidget):
    def __init__(self, memory_manager=None, semantic_config=None, parent=None):
        super().__init__(parent)
        self.memory_manager = memory_manager or OptimizedMemoryManager()
        self.semantic_config = semantic_config or SemanticSearchConfig()
        self.setup_ui()
        

        self.max_memory_percent.valueChanged.connect(self.update_memory_settings)
        self.max_cache_size.valueChanged.connect(self.update_memory_settings)
        self.emergency_threshold.valueChanged.connect(self.update_memory_settings)
        self.optimization_interval.valueChanged.connect(self.update_memory_settings)
        self.model_type.currentTextChanged.connect(self.update_semantic_settings)
        self.batch_size.valueChanged.connect(self.update_semantic_settings)
        self.similarity_threshold.valueChanged.connect(self.update_semantic_settings)
        self.use_gpu.stateChanged.connect(self.update_semantic_settings)
        

        self.memory_timer = QTimer()
        self.memory_timer.timeout.connect(self.update_memory_stats)
        self.memory_timer.start(1000)

    def setup_ui(self):
        layout = QVBoxLayout()


        memory_group = QGroupBox("Memory Management Settings")
        memory_layout = QFormLayout()


        self.max_memory_percent = QDoubleSpinBox()
        self.max_memory_percent.setRange(10.0, 95.0)
        self.max_memory_percent.setSingleStep(5.0)
        self.max_memory_percent.setValue(80.0)
        memory_layout.addRow("Max Memory Percent:", self.max_memory_percent)


        self.max_cache_size = QSpinBox()
        self.max_cache_size.setRange(64, 8192)
        self.max_cache_size.setSingleStep(64)
        self.max_cache_size.setValue(512)
        memory_layout.addRow("Max Cache Size (MB):", self.max_cache_size)


        self.emergency_threshold = QDoubleSpinBox()
        self.emergency_threshold.setRange(80.0, 99.0)
        self.emergency_threshold.setSingleStep(1.0)
        self.emergency_threshold.setValue(95.0)
        memory_layout.addRow("Emergency Threshold (%):", self.emergency_threshold)


        self.optimization_interval = QSpinBox()
        self.optimization_interval.setRange(5, 300)
        self.optimization_interval.setSingleStep(5)
        self.optimization_interval.setValue(30)
        memory_layout.addRow("Optimization Interval (sec):", self.optimization_interval)


        self.memory_stats_label = QLabel("Memory Statistics: Loading...")
        memory_layout.addRow("", self.memory_stats_label)


        self.emergency_clear_btn = QPushButton("Emergency Memory Clear")
        self.emergency_clear_btn.clicked.connect(self.force_emergency_clear)
        memory_layout.addRow("", self.emergency_clear_btn)

        memory_group.setLayout(memory_layout)
        layout.addWidget(memory_group)


        indexer_group = QGroupBox("Indexer Settings")
        indexer_layout = QFormLayout()


        self.indexer_type_combo = QComboBox()
        self.indexer_type_combo.addItems(["Auto", "Standard", "Optimized"])
        self.indexer_type_combo.setCurrentText("Auto")
        self.indexer_type_combo.setToolTip("Auto: Automatic detection\nStandard: Basic indexing\nOptimized: Advanced indexing with memory management")
        indexer_layout.addRow("Default Indexer Type:", self.indexer_type_combo)


        index_file_layout = QHBoxLayout()
        self.index_file_btn = QPushButton("Select Index File")
        self.index_file_path = QLineEdit()
        self.index_file_path.setReadOnly(True)
        self.index_file_path.setPlaceholderText("eiron_index.bin")
        index_file_layout.addWidget(self.index_file_btn)
        index_file_layout.addWidget(self.index_file_path)
        indexer_layout.addRow("Default Index File:", index_file_layout)


        self.index_status = QLabel("No index found")
        indexer_layout.addRow("Index Status:", self.index_status)


        self.index_file_btn.clicked.connect(self.select_index_file)
        self.indexer_type_combo.currentTextChanged.connect(self.update_indexer_settings)

        indexer_group.setLayout(indexer_layout)
        layout.addWidget(indexer_group)


        semantic_group = QGroupBox("Semantic Search Settings")
        semantic_layout = QFormLayout()

        self.model_type = QComboBox()
        self.model_type.addItems(self.semantic_config.get_available_models())
        self.model_type.currentTextChanged.connect(self.update_semantic_settings)
        semantic_layout.addRow("Model Type:", self.model_type)

        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 128)
        self.batch_size.setValue(self.semantic_config.batch_size)
        self.batch_size.valueChanged.connect(self.update_semantic_settings)
        semantic_layout.addRow("Batch Size:", self.batch_size)

        self.similarity_threshold = QDoubleSpinBox()
        self.similarity_threshold.setRange(0.0, 1.0)
        self.similarity_threshold.setSingleStep(0.1)
        self.similarity_threshold.setValue(self.semantic_config.similarity_threshold)
        self.similarity_threshold.valueChanged.connect(self.update_semantic_settings)
        semantic_layout.addRow("Similarity Threshold:", self.similarity_threshold)

        self.use_gpu = QCheckBox()
        self.use_gpu.setChecked(self.semantic_config.use_gpu)
        self.use_gpu.stateChanged.connect(self.update_semantic_settings)
        semantic_layout.addRow("Use GPU:", self.use_gpu)

        semantic_group.setLayout(semantic_layout)
        layout.addWidget(semantic_group)

        self.setLayout(layout)

    def update_memory_settings(self):
        
        try:

            self.memory_manager = OptimizedMemoryManager(
                max_memory_percent=self.max_memory_percent.value(),
                max_cache_size_mb=self.max_cache_size.value(),
                emergency_threshold=self.emergency_threshold.value(),
                optimization_interval=self.optimization_interval.value()
            )
            logging.info("Memory settings updated")
        except Exception as e:
            logging.error(f"Error updating memory settings: {e}")

    def update_semantic_settings(self):
        
        try:
            self.semantic_config.model_type = self.model_type.currentText()
            self.semantic_config.batch_size = self.batch_size.value()
            self.semantic_config.similarity_threshold = self.similarity_threshold.value()
            self.semantic_config.use_gpu = self.use_gpu.isChecked()
            logging.info("Semantic settings updated")
        except Exception as e:
            logging.error(f"Error updating semantic settings: {e}")

    def force_emergency_clear(self):
        
        try:
            if self.memory_manager:
                self.memory_manager.emergency_clear()
                logging.info("Emergency memory clear executed")
        except Exception as e:
            logging.error(f"Error during emergency clear: {e}")

    def update_memory_stats(self):
        
        try:
            if self.memory_manager:
                stats = self.memory_manager.get_stats()
                detailed_stats = self.memory_manager.get_detailed_stats()
                
                stats_text = (
                    f"Process: {stats.process_memory_mb:.1f}MB | "
                    f"Cache: {stats.cache_size_mb:.1f}MB | "
                    f"Items: {stats.cache_items} | "
                    f"Optimizations: {detailed_stats['performance']['optimization_count']} | "
                    f"Emergency: {detailed_stats['performance']['emergency_clear_count']}"
                )
                self.memory_stats_label.setText(stats_text)
        except Exception as e:
            self.memory_stats_label.setText(f"Error loading stats: {e}")

    def select_index_file(self):
        
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Index File",
            "",
            "Index Files (eiron_index.bin);;All Files (*.*)"
        )
        if file:
            self.index_file_path.setText(file)
            self.check_custom_index(file)

    def check_custom_index(self, index_path):
        
        try:
            from core.indexer_factory import create_auto_indexer, get_auto_indexer_info
            
            if os.path.exists(index_path):

                index_info = get_auto_indexer_info(index_path)
                
                if index_info['exists'] and index_info['total_docs'] > 0:

                    indexer_type = index_info.get('recommended_indexer', 'standard').title()
                    confidence = index_info.get('confidence', 0.0)
                    

                    file_size_mb = index_info.get('file_size', 0) / 1024 / 1024
                    metadata_size_mb = index_info.get('metadata_size', 0) / 1024 / 1024
                    db_size_mb = index_info.get('db_size', 0) / 1024 / 1024
                    total_size_mb = index_info.get('total_size', 0) / 1024 / 1024
                    

                    files_info = f"Main: {file_size_mb:.1f}MB"
                    if metadata_size_mb > 0:
                        files_info += f", Meta: {metadata_size_mb:.1f}MB"
                    if db_size_mb > 0:
                        files_info += f", DB: {db_size_mb:.1f}MB"
                    files_info += f" (Total: {total_size_mb:.1f}MB)"
                    
                    status_text = (
                        f"Found pre-indexed files: {index_info['total_docs']} documents\n"
                        f"Auto-detected type: {indexer_type} (confidence: {confidence:.1f})\n"
                        f"Total terms: {index_info.get('total_terms', 0)}\n"
                        f"Files: {files_info}"
                    )
                    
                    self.index_status.setText(status_text)
                    logging.info(f"Loaded custom index file: {index_info['total_docs']} documents "
                               f"with {indexer_type} indexer")
                else:
                    self.index_status.setText("Index file is empty or invalid")
            else:
                self.index_status.setText("Index file not found")
        except Exception as e:
            logging.error(f"Error loading custom index: {e}")
            self.index_status.setText("Error loading index file")

    def update_indexer_settings(self):
        
        indexer_type = self.indexer_type_combo.currentText()
        index_file = self.index_file_path.text() or 'eiron_index.bin'
        
        logging.info(f"Indexer settings updated: type={indexer_type}, file={index_file}")
        

        if os.path.exists(index_file):
            self.check_custom_index(index_file)

    def get_indexer_settings(self):
        
        return {
            'type': self.indexer_type_combo.currentText().lower(),
            'file': self.index_file_path.text() or 'eiron_index.bin'
        }

class SearchTab(QWidget):
    def __init__(self, memory_manager=None, semantic_config=None, settings_tab=None, parent=None):
        super().__init__(parent)
        self.memory_manager = memory_manager
        self.semantic_config = semantic_config
        self.settings_tab = settings_tab
        self._results_buffer = []
        self.setup_ui()
        

        try:
            from core.indexer_factory import create_auto_indexer, get_auto_indexer_info
            self.indexer = create_auto_indexer('eiron_index.bin')
            self.indexer_info = get_auto_indexer_info('eiron_index.bin')
            logging.info(f"Auto-detected indexer: {self.indexer_info.get('recommended_indexer', 'standard')}")
        except Exception as e:
            logging.error(f"Error creating auto-indexer: {e}")

            from core.indexer import Indexer
            self.indexer = Indexer()
            self.indexer_info = {'recommended_indexer': 'standard', 'confidence': 0.0}
            logging.info("Using fallback standard indexer")
        self.searcher = None
        

        self.index_btn.clicked.connect(self.start_indexing)
        self.search_btn.clicked.connect(self.start_search)
        self.folder_btn.clicked.connect(self.select_folder)
        

        self.check_index_exists()

    def check_index_exists(self):
        

        index_file = self.get_index_file_from_settings()
        index_exists = os.path.exists(index_file)
        self.search_btn.setEnabled(index_exists)
        if not index_exists:
            self.search_btn.setToolTip("Please index files first")
            self.index_status.setText("No index found")
        else:
            self.search_btn.setToolTip("")
            self.load_index_info()

    def get_index_file_from_settings(self):
        
        if self.settings_tab and hasattr(self.settings_tab, 'get_indexer_settings'):
            settings = self.settings_tab.get_indexer_settings()
            return settings.get('file', 'eiron_index.bin')
        return 'eiron_index.bin'

    def get_indexer_type_from_settings(self):
        
        if self.settings_tab and hasattr(self.settings_tab, 'get_indexer_settings'):
            settings = self.settings_tab.get_indexer_settings()
            return settings.get('type', 'auto')
        return 'auto'

    def load_index_info(self):
        
        try:
            from core.indexer_factory import create_auto_indexer, get_auto_indexer_info
            

            index_file = self.get_index_file_from_settings()
            

            index_info = get_auto_indexer_info(index_file)
            
            if index_info['exists'] and index_info['total_docs'] > 0:

                self.indexer = create_auto_indexer(index_file)
                self.indexer_info = index_info
                

                self.index_file_path.setText(index_file)
                

                indexer_type = index_info.get('recommended_indexer', 'standard').title()
                confidence = index_info.get('confidence', 0.0)
                

                file_size_mb = index_info.get('file_size', 0) / 1024 / 1024
                metadata_size_mb = index_info.get('metadata_size', 0) / 1024 / 1024
                db_size_mb = index_info.get('db_size', 0) / 1024 / 1024
                total_size_mb = index_info.get('total_size', 0) / 1024 / 1024
                

                files_info = f"Main: {file_size_mb:.1f}MB"
                if metadata_size_mb > 0:
                    files_info += f", Meta: {metadata_size_mb:.1f}MB"
                if db_size_mb > 0:
                    files_info += f", DB: {db_size_mb:.1f}MB"
                files_info += f" (Total: {total_size_mb:.1f}MB)"
                
                status_text = (
                    f"Found pre-indexed files: {index_info['total_docs']} documents\n"
                    f"Auto-detected type: {indexer_type} (confidence: {confidence:.1f})\n"
                    f"Total terms: {index_info.get('total_terms', 0)}\n"
                    f"Files: {files_info}"
                )
                
                self.index_status.setText(status_text)
                logging.info(f"Loaded pre-indexed files: {index_info['total_docs']} documents "
                           f"with {indexer_type} indexer")
            else:
                self.index_status.setText("No valid index found")
                logging.info("No valid index found")
                
        except Exception as e:
            logging.error(f"Error loading index info: {e}")
            self.index_status.setText("Error loading index info")

    def add_search_result(self, result):
        
        self._results_buffer.append(result)
        self._flush_results()

    def _flush_results(self):
        
        if self._results_buffer:
            self.results_list.append('\n'.join(self._results_buffer))
            self._results_buffer.clear()

    def setup_ui(self):
        layout = QVBoxLayout()
        

        folder_layout = QHBoxLayout()
        self.folder_btn = QPushButton("Select Folder")
        self.folder_path = QLineEdit()
        self.folder_path.setReadOnly(True)
        folder_layout.addWidget(self.folder_btn)
        folder_layout.addWidget(self.folder_path)
        layout.addLayout(folder_layout)
        

        index_file_layout = QHBoxLayout()
        index_file_layout.addWidget(QLabel("Index File:"))
        self.index_file_path = QLineEdit()
        self.index_file_path.setReadOnly(True)
        self.index_file_path.setPlaceholderText("eiron_index.bin")
        index_file_layout.addWidget(self.index_file_path)
        index_file_layout.addStretch()
        layout.addLayout(index_file_layout)
        

        self.index_status = QLabel("No index found")
        layout.addWidget(self.index_status)
        

        button_layout = QHBoxLayout()
        self.index_btn = QPushButton("Index")
        self.search_btn = QPushButton("Search")
        self.search_btn.setEnabled(False)
        button_layout.addWidget(self.index_btn)
        button_layout.addWidget(self.search_btn)
        layout.addLayout(button_layout)
        

        self.index_progress = QProgressBar()
        self.index_progress.setVisible(False)
        layout.addWidget(self.index_progress)
        

        search_options = QHBoxLayout()
        

        search_options.addWidget(QLabel("Search Mode:"))
        self.search_mode_combo = QComboBox()
        self.search_mode_combo.addItems(["hybrid", "keyword", "semantic", "exact"])
        self.search_mode_combo.setCurrentText("hybrid")
        search_options.addWidget(self.search_mode_combo)
        

        self.similarity_label = QLabel("Similarity Threshold:")
        self.similarity_slider = QDoubleSpinBox()
        self.similarity_slider.setRange(0.1, 1.0)
        self.similarity_slider.setSingleStep(0.1)
        self.similarity_slider.setValue(0.6)
        search_options.addWidget(self.similarity_label)
        search_options.addWidget(self.similarity_slider)
        

        self.hide_paths_cb = QCheckBox("Hide Paths")
        search_options.addWidget(self.hide_paths_cb)
        
        self.dynamic_output_cb = QCheckBox("Dynamic Output")
        self.dynamic_output_cb.setChecked(True)
        search_options.addWidget(self.dynamic_output_cb)
        
        self.enable_rlimit_cb = QCheckBox("Enable RLIMIT")
        search_options.addWidget(self.enable_rlimit_cb)
        
        layout.addLayout(search_options)
        

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter search query...")
        layout.addWidget(self.search_input)
        

        self.results_list = QTextEdit()
        self.results_list.setReadOnly(True)
        layout.addWidget(self.results_list)
        
        self.setLayout(layout)

    def select_folder(self):
        
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.folder_path.setText(folder)
            logging.info(f"Selected directory: {folder}")



    def start_indexing(self):
        
        path = self.folder_path.text()
        if not path:
            QMessageBox.warning(self, "Warning", "Please select a folder first")
            return


        indexer_type = self.get_indexer_type_from_settings()
        index_file = self.get_index_file_from_settings()


        resource_limits = None
        if self.enable_rlimit_cb.isChecked():
            resource_limits = {
                'max_memory_mb': 1024,
                'max_cpu_percent': 80.0,
                'max_file_size_mb': 100,
                'max_open_files': 1024,
                'enable_rlimit': True
            }
        

        try:
            from core.indexer_factory import create_auto_indexer
            

            force_type = None if indexer_type == 'auto' else indexer_type
            

            indexer_params = {}
            if indexer_type == 'standard':
                indexer_params.update({
                    'hide_paths': self.hide_paths_cb.isChecked(),
                    'dynamic_output': self.dynamic_output_cb.isChecked(),
                    'resource_limits': resource_limits
                })
            elif indexer_type == 'optimized':
                indexer_params.update({
                    'max_workers': 8,
                    'batch_size': 50
                })
            
            self.indexer = create_auto_indexer(
                index_file=index_file,
                force_type=force_type,
                **indexer_params
            )
            
            indexer_type_name = force_type or "auto-detected"
            logging.info(f"Created {indexer_type_name} indexer")
            
        except Exception as e:
            logging.error(f"Error creating indexer: {e}")
            QMessageBox.critical(self, "Error", f"Failed to create indexer: {str(e)}")
            return

        self.index_btn.setEnabled(False)
        self.index_status.setText(f"Indexing with {indexer_type} indexer...")
        self.index_progress.setVisible(True)
        self.index_progress.setRange(0, 0)
        

        self.index_thread = IndexThread(path, indexer_type, index_file, self.memory_manager)
        self.index_thread.finished.connect(self.indexing_finished)
        self.index_thread.error.connect(self.indexing_error)
        self.index_thread.log.connect(lambda msg: logging.info(f"Indexing: {msg}"))
        self.index_thread.start()
        logging.info(f"Starting indexing process for: {path} with {indexer_type} indexer")

    def indexing_finished(self):
        
        self.index_btn.setEnabled(True)
        self.index_progress.setVisible(False)
        self.check_index_exists()
        logging.info("Indexing completed successfully")
        QMessageBox.information(self, "Success", "Indexing completed successfully")

    def indexing_error(self, error_msg):
        
        self.index_btn.setEnabled(True)
        self.index_progress.setVisible(False)
        self.index_status.setText("Indexing failed")
        logging.error(f"Indexing failed: {error_msg}")
        QMessageBox.critical(self, "Error", f"Indexing failed: {error_msg}")

    def start_search(self):
        
        query = self.search_input.text().strip()
        if not query:
            QMessageBox.warning(self, "Warning", "Please enter a search query")
            return

        self.search_btn.setEnabled(False)
        self.results_list.clear()
        self._results_buffer.clear()
        
        try:
            search_indexer = self.indexer
            
            search_mode = self.search_mode_combo.currentText()
            logging.info(f"Starting search: '{query}' (mode: {search_mode}, indexer: auto)")
            
            if hasattr(search_indexer, 'search'):
                results = search_indexer.search(
                    query,
                    search_mode=search_mode,
                    max_results=20,
                    min_similarity=self.similarity_slider.value()
                )
            else:
                from core.context_search import ContextSearchEngine
                search_engine = ContextSearchEngine(search_indexer)
                search_results = search_engine.search(
                    query,
                    max_results=20
                )
                results = [(result.doc_id, result.relevance_score, result.context) for result in search_results]
            
            for doc_id, score, context in results:
                file_path = doc_id  # fallback
                
                try:
                    if hasattr(search_indexer, 'index_data') and isinstance(search_indexer.index_data, dict):
                        # OptimizedIndexer
                        doc_data = search_indexer.index_data.get('documents', {}).get(doc_id, {})
                        file_path = doc_data.get('file_path', doc_id)
                    elif hasattr(search_indexer, 'index') and hasattr(search_indexer.index, 'documents'):
                        # Standard Indexer
                        doc_data = search_indexer.index.documents.get(doc_id, {})
                        file_path = doc_data.get('metadata', {}).get('original_path', doc_id)
                except Exception:
                    file_path = doc_id
                
                if self.hide_paths_cb.isChecked():
                    filename = os.path.basename(file_path) if file_path != doc_id else doc_id
                    display_path = f"{filename} ({doc_id[:8]})"
                else:
                    display_path = file_path
                
                result_text = f"File: {display_path}\n"
                result_text += f"Relevance: {score:.3f}\n"
                if context:
                    result_text += f"Context: {context[:200]}...\n" if len(context) > 200 else f"Context: {context}\n"
                result_text += f"{'='*50}\n"
                
                self.add_search_result(result_text)
            
            logging.info(f"Search completed: found {len(results)} results")
            self.search_finished()
            
        except Exception as e:
            logging.error(f"Search failed: {str(e)}")
            self.search_error(str(e))

    def search_finished(self):
        
        self.search_btn.setEnabled(True)
        if self.results_list.toPlainText() == "":
            QMessageBox.information(self, "Search", "No results found")

    def search_error(self, error_msg):
        
        self.search_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Search failed: {error_msg}")

class EironGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eiron Search")
        self.setGeometry(100, 100, 1200, 800)
        

        self.memory_manager = MemoryManager()
        self.semantic_config = SemanticSearchConfig()
        

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        

        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        

        self.settings_tab = SettingsTab(memory_manager=self.memory_manager, semantic_config=self.semantic_config)
        self.search_tab = SearchTab(memory_manager=self.memory_manager, semantic_config=self.semantic_config, settings_tab=self.settings_tab)
        

        self.tab_widget.addTab(self.search_tab, "Search")
        self.tab_widget.addTab(self.settings_tab, "Settings")
        

        log_group = QGroupBox("Application Logs")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        self.log_text.setStyleSheet("background-color: #f0f0f0;")
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        

        self.setup_logging()
        

        self.memory_timer = QTimer()
        self.memory_timer.timeout.connect(self.update_memory_stats)
        self.memory_timer.start(1000)
        

        self.search_tab.check_index_exists()

    def setup_logging(self):
        
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        handler = LogHandler(self.log_text)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        logging.info("EIRON Enhanced Context Search System started")
        logging.info("Ready for indexing and searching documents")
        logging.info("Available search modes: hybrid, keyword, semantic, exact")

    def update_memory_stats(self):
        
        self.settings_tab.update_memory_stats()

if __name__ == "__main__":
    app = QApplication([])
    window = EironGUI()
    window.show()
    sys.exit(app.exec())
