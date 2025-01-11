import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QTableView, QVBoxLayout, QWidget, QTabWidget, QPushButton, QTextEdit, QLabel, QListWidget, QHBoxLayout
)
from PyQt5.QtCore import QAbstractTableModel, Qt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

class PandasModel(QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self._data.columns[section]
            elif orientation == Qt.Vertical:
                return str(section)
        return None

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Analizador de Archivos: Malicious y NoDoH")
        self.setGeometry(100, 100, 1200, 800)

        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        self.load_tab = QWidget()
        self.tab_widget.addTab(self.load_tab, "Carga de Datos")
        self.load_layout = QVBoxLayout(self.load_tab)

        self.doh_label = QLabel("TRAFICO DOH")
        self.nodoh_label = QLabel("TRAFICO NODOH")

        self.btn_malicious = QPushButton("Cargar archivo Malicious")
        self.btn_malicious.clicked.connect(self.load_malicious_file)

        self.btn_nodoh = QPushButton("Cargar archivo NoDoH")
        self.btn_nodoh.clicked.connect(self.load_nodoh_file)

        self.load_layout.addWidget(self.btn_malicious)
        self.load_layout.addWidget(self.btn_nodoh)

        self.table_malicious = QTableView()
        self.table_nodoh = QTableView()

        self.load_layout.addWidget(self.doh_label)
        self.load_layout.addWidget(self.table_malicious)
        self.load_layout.addWidget(self.nodoh_label)
        self.load_layout.addWidget(self.table_nodoh)

        self.results_tab = QWidget()
        self.tab_widget.addTab(self.results_tab, "Resultados")
        self.results_layout = QVBoxLayout(self.results_tab)

        self.analysis_list = QListWidget()
        self.results_layout.addWidget(self.analysis_list)

        self.btn_run_model = QPushButton("Ejecutar Modelo XGBClassifier")
        self.btn_run_model.clicked.connect(self.run_model)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_layout.addWidget(self.btn_run_model)
        self.results_layout.addWidget(self.results_text)

        self.graphs_tab = QWidget()
        self.tab_widget.addTab(self.graphs_tab, "Gráficas")
        self.graphs_layout = QVBoxLayout(self.graphs_tab)

        self.graph_buttons_layout = QHBoxLayout()

        self.btn_class_distribution = QPushButton("Distribución de Clases")
        self.btn_class_distribution.clicked.connect(self.plot_class_distribution)
        self.graph_buttons_layout.addWidget(self.btn_class_distribution)

        self.btn_timestamp_graph = QPushButton("Gráfica TimeStamp")
        self.btn_timestamp_graph.clicked.connect(self.plot_timestamp_graph)
        self.graph_buttons_layout.addWidget(self.btn_timestamp_graph)

        self.graphs_layout.addLayout(self.graph_buttons_layout)
        self.graph_canvas = FigureCanvas(plt.figure())
        self.graphs_layout.addWidget(self.graph_canvas)

        self.malicious_data = None
        self.nodoh_data = None

    def load_malicious_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Abrir archivo Malicious", "", "Archivos CSV (*.csv)")
        if file_path:
            self.malicious_data = pd.read_csv(file_path)
            self.display_data(self.malicious_data, self.table_malicious)

    def load_nodoh_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Abrir archivo NoDoH", "", "Archivos CSV (*.csv)")
        if file_path:
            self.nodoh_data = pd.read_csv(file_path)
            self.display_data(self.nodoh_data, self.table_nodoh)

    def display_data(self, data, table_view):
        model = PandasModel(data)
        table_view.setModel(model)

    def run_model(self):
        if self.malicious_data is not None and self.nodoh_data is not None:
            combined_df = pd.concat([self.malicious_data, self.nodoh_data], ignore_index=True)
            combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

            combined_df = combined_df.replace([float('inf'), -float('inf')], None)
            combined_df = combined_df.dropna()

            X = combined_df.drop(columns=['Label', 'SourceIP', 'DestinationIP', 'SourcePort', 'DestinationPort', 'TimeStamp'])
            y = combined_df['Label']

            scaler = MinMaxScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
            xgb_model.fit(X_train, y_train)

            y_pred = xgb_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            scores = cross_val_score(xgb_model, X, y, cv=5, scoring='accuracy')

            analysis_number = self.analysis_list.count() + 1
            self.analysis_list.addItem(f"Análisis {analysis_number}")

            self.results_text.setText(f"Accuracy: {accuracy:.4f}\n\nClassification Report:\n{report}\n\nScores por fold: {scores}\nPrecisión promedio: {scores.mean():.4f}\nDesviación estándar: {scores.std():.4f}")

    def plot_class_distribution(self):
        if self.malicious_data is not None and self.nodoh_data is not None:
            plt.clf()
            combined_df = pd.concat([self.malicious_data, self.nodoh_data], ignore_index=True)
            ax = self.graph_canvas.figure.add_subplot(111)
            combined_df['Label'].value_counts().plot(kind='bar', ax=ax, color=['blue', 'orange'])
            ax.set_title('Distribución de Clases (DoH vs NoDoH)')
            ax.set_xlabel('Clase')
            ax.set_ylabel('Frecuencia')
            self.graph_canvas.draw()

    def plot_timestamp_graph(self):
        if self.malicious_data is not None and self.nodoh_data is not None:
            plt.clf()
            combined_df = pd.concat([self.malicious_data, self.nodoh_data], ignore_index=True)
            ax = self.graph_canvas.figure.add_subplot(111)
            combined_df.plot(x='TimeStamp', y='FlowBytesSent', ax=ax, color='green', legend=False)
            ax.set_title('Flujo de Bytes Enviados vs TimeStamp')
            ax.set_xlabel('TimeStamp')
            ax.set_ylabel('FlowBytesSent')
            self.graph_canvas.draw()

# Función principal
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
