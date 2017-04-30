from PyQt5 import QtCore, QtGui, QtWidgets
import random, sys, warnings
from PyQt5.QtWidgets import (QVBoxLayout, QGridLayout,QMainWindow,QApplication)
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
import matplotlib.pyplot as plt

warnings.simplefilter("ignore", DeprecationWarning)

class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi()

    def setupUi(self):

        #Initilize variables for tutorial box pages
        self.current_page, self.max_Page = 0, 0
        self.app = 1

        #Size window to default dimensions
        self.resize(1800, 900)
        self.setWindowTitle("MaLeTS")

        #Create a central Widget
        self.centralwidget = QtWidgets.QWidget(self)

        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)

        self.gridLayout = QGridLayout()
        self.inputBox = QtWidgets.QGroupBox(self.centralwidget, title="Input")
        self.inputBox.setAlignment(QtCore.Qt.AlignCenter)

        self.dataTypeGroupBox = QtWidgets.QGroupBox()
        self.dataTypeLabel = QtWidgets.QLabel("Data Type:")
        self.dataSelectionComboBox = QtWidgets.QComboBox()
        self.dataSelectionComboBox.addItem("Numerical")
        self.dataGroupBox = QtWidgets.QStackedWidget()
        self.dataGroupBox.setInputMethodHints(QtCore.Qt.ImhNone)
        self.dataGroupBox.setFrameShape(QtWidgets.QFrame.Box)

        self.numericalPage = QtWidgets.QWidget()
        self.dataBox = QtWidgets.QGroupBox()
        self.generateNumbersGroupBox = QtWidgets.QGroupBox()
        self.generateNumbersLabel = QtWidgets.QLabel("(# of Dims, # of Points, # of Labels, Min #, Max #):")
        self.generateNumbersEdit = QtWidgets.QLineEdit()

        self.dataLabel = QtWidgets.QLabel("Data:")
        self.dataPlainTextEdit = QtWidgets.QPlainTextEdit()
        self.labelsBox = QtWidgets.QGroupBox()

        self.labelsLabel = QtWidgets.QLabel("Labels:")
        self.labelsPlainTextEdit = QtWidgets.QPlainTextEdit()
        self.generateButton = QtWidgets.QPushButton("Generate Numbers")
        self.generateButton.clicked.connect(self.generateNumbers)
        self.dataGroupBox.addWidget(self.numericalPage)
        self.inputVerticalLayout = QtWidgets.QVBoxLayout(self.inputBox)
        self.inputVerticalLayout.addWidget(self.dataTypeGroupBox)
        self.inputVerticalLayout.addWidget(self.dataGroupBox)
        self.dataTypeGroupBoxHorizontalLayout = QtWidgets.QHBoxLayout(self.dataTypeGroupBox)
        self.dataTypeGroupBoxHorizontalLayout.addWidget(self.dataTypeLabel)
        self.dataTypeGroupBoxHorizontalLayout.addWidget(self.dataSelectionComboBox)
        self.numericalPageVerticalLayout = QtWidgets.QVBoxLayout(self.numericalPage)
        self.numericalPageVerticalLayout.addWidget(self.dataBox)
        self.numericalPageVerticalLayout.addWidget(self.labelsBox)
        self.numericalPageVerticalLayout.addWidget(self.generateNumbersGroupBox)
        self.generateNumbersGroupBoxVerticalLayout = QtWidgets.QVBoxLayout(self.generateNumbersGroupBox)
        self.generateNumbersGroupBoxVerticalLayout.addWidget(self.generateNumbersLabel)
        self.generateNumbersGroupBoxVerticalLayout.addWidget(self.generateNumbersEdit)
        self.generateNumbersGroupBoxVerticalLayout.addWidget(self.generateButton)
        self.dataBoxHorizontalLayout = QtWidgets.QHBoxLayout(self.dataBox)
        self.dataBoxHorizontalLayout.addWidget(self.dataLabel)
        self.dataBoxHorizontalLayout.addWidget(self.dataPlainTextEdit)
        self.labelBoxHorizontalLayout = QtWidgets.QHBoxLayout(self.labelsBox)
        self.labelBoxHorizontalLayout.addWidget(self.labelsLabel)
        self.labelBoxHorizontalLayout.addWidget(self.labelsPlainTextEdit)

        self.algorithmBox = QtWidgets.QGroupBox(self.centralwidget, title="Algorithm")
        self.algorithmBox.setAlignment(QtCore.Qt.AlignCenter)
        self.algorithmSelectionGroupBox = QtWidgets.QGroupBox()
        self.algorithmSelectionGroupBox.setTitle("")
        self.algorithmLabel = QtWidgets.QLabel("Algorithm:")
        self.algorithmSelectionBox = QtWidgets.QComboBox()
        self.algorithmSelectionBox.addItem("Linear Support Vector Classifier")
        self.algorithmSelectionBox.addItem("K Nearest Neighbors Classifier")
        self.algorithmSelectionBox.addItem("Decision Tree Classifier")
        self.parametersBox = QtWidgets.QGroupBox("Parameters")
        self.parametersBox.setGeometry(QtCore.QRect(0, 40, 275, 270))
        self.parametersBox2 = QtWidgets.QGroupBox("Parameters2")
        self.algorithmBoxVerticalLayout = QtWidgets.QVBoxLayout(self.algorithmBox)
        self.algorithmBoxVerticalLayout.addWidget(self.algorithmSelectionGroupBox)
        self.algorithmBoxVerticalLayout.addWidget(self.parametersBox)
        self.algorithmSelectionGroupBoxHorizontalLayout = QtWidgets.QVBoxLayout(self.algorithmSelectionGroupBox)
        self.algorithmSelectionGroupBoxHorizontalLayout.addWidget(self.algorithmLabel)
        self.algorithmSelectionGroupBoxHorizontalLayout.addWidget(self.algorithmSelectionBox)

        self.outputBox = QtWidgets.QGroupBox(self.centralwidget, title="Output")
        self.outputBox.setAlignment(QtCore.Qt.AlignCenter)

        self.outputWidget = QtWidgets.QWidget(self.outputBox)
        self.plotBox = QtWidgets.QGroupBox(self.centralwidget, title="Plot")
        self.plotBox.setAlignment(QtCore.Qt.AlignCenter)
        fig = plt.figure(1, figsize=(1, 1),dpi=50)
        self.canvas = FigureCanvas(fig)
        self.canvas.draw()
        self.hbox3 = QVBoxLayout(self.plotBox)

        self.outputPlainTextEdit = QtWidgets.QPlainTextEdit()
        self.outputPredictLabel = QtWidgets.QLabel()
        self.outputPredictLabel.setText("Predict:")
        self.outputPredictLineEdit = QtWidgets.QLineEdit()
        self.outputVerticalLayout = QtWidgets.QVBoxLayout(self.outputBox)
        self.outputVerticalLayout.addWidget(self.outputPlainTextEdit)
        self.outputVerticalLayout.addWidget(self.outputPredictLabel)
        self.outputVerticalLayout.addWidget(self.outputPredictLineEdit)


        self.tutorialBox = QtWidgets.QGroupBox(self.centralwidget, title="Tutorial")
        self.tutorialBox.setAlignment(QtCore.Qt.AlignCenter)
        self.tutorialPlainTextEdit = QtWidgets.QPlainTextEdit()

        self.previousButton = QtWidgets.QPushButton('Previous Page')
        self.previousButton.clicked.connect(self.previous_Page)
        self.nextButton = QtWidgets.QPushButton('Next Page')
        self.nextButton.clicked.connect(self.next_Page)

        self.tutorialGroupBox = QtWidgets.QGroupBox()
        self.tutorialHorizontalLayout = QtWidgets.QHBoxLayout(self.tutorialGroupBox)
        self.tutorialHorizontalLayout.addWidget(self.previousButton)
        self.tutorialHorizontalLayout.addWidget(self.nextButton)
        self.tutorialVerticalLayout = QtWidgets.QVBoxLayout(self.tutorialBox)
        self.tutorialVerticalLayout.addWidget(self.tutorialPlainTextEdit)
        self.tutorialVerticalLayout.addWidget(self.tutorialGroupBox)
        self.plotVerticalLayout = QtWidgets.QVBoxLayout(self.plotBox)
        self.plotVerticalLayout.addWidget(self.canvas)
        self.scroll = QtWidgets.QScrollArea(self.parametersBox)
        self.scrollAreaContents = QtWidgets.QWidget(self.scroll)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.scrollAreaContents)
        self.hbox2 = QVBoxLayout(self.parametersBox)
        self.hbox2.addWidget(self.scroll)
        self.hbox4 = QVBoxLayout(self.scroll)
        self.hbox4.addWidget(self.scrollAreaContents)
        self.parametersVerticalLayout = QtWidgets.QVBoxLayout(self.scrollAreaContents)
        menubar = self.menuBar()

        for letter in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']:
            exec('self.%s =QtWidgets.QLineEdit()' % letter)
            exec('self.%sl = QtWidgets.QLabel()' % letter)
            exec('self.%sBox = QtWidgets.QGroupBox(self.scrollAreaContents)' % letter)
            exec('self.%sHorizontalLayout = QtWidgets.QHBoxLayout(self.%sBox)' % (letter, letter))
            exec('self.%sHorizontalLayout.addWidget(self.%sl)' % (letter, letter))
            exec('self.%sHorizontalLayout.addWidget(self.%s)' % (letter, letter))
            exec('self.parametersVerticalLayout.addWidget(self.%sBox)' % letter)
            exec('self.%sBox.hide()' % letter)

        font = QtGui.QFont()
        font.setFamily("Segoe UI Historic")
        font.setWeight(50)
        menubar.setFont(font)
        menuFile = QtWidgets.QMenu(menubar)
        menuFile.setObjectName("menuFile")

        actionOpen = QtWidgets.QAction(self)
        actionOpen.setObjectName("actionOpen")
        actionOpen.setShortcut("Ctrl+O")
        actionOpen.setStatusTip('Open File')
        actionOpen.triggered.connect(self.file_open)
        actionSave = QtWidgets.QAction(self)
        actionSave.setObjectName("actionSave")
        actionSave.setShortcut("Ctrl+S")
        actionSave.setStatusTip('Save File')
        actionSave.triggered.connect(self.file_save)
        actionExit = QtWidgets.QAction(self)
        actionExit.setShortcut('Ctrl+Q')
        actionExit.triggered.connect(self.close)
        menuFile.addAction(actionOpen)
        menuFile.addAction(actionSave)
        menuFile.addAction(actionExit)
        menuFile.setTitle("File")
        actionOpen.setText("Open")
        actionSave.setText("Save")
        actionExit.setText("Exit")
        menubar.addAction(menuFile.menuAction())
        self.gridLayout.addWidget(self.inputBox,0,0,1,1)
        self.gridLayout.addWidget(self.algorithmBox,0,1,1,1)
        self.gridLayout.addWidget(self.outputBox,0,2,1,1)
        self.gridLayout.addWidget(self.tutorialBox,1,0,1,2)
        self.gridLayout.addWidget(self.plotBox, 1, 2, 1, 1)
        self.setCentralWidget(self.centralwidget)
        self.verticalLayout_2.addLayout(self.gridLayout)
        self.centralwidget.setLayout(self.verticalLayout_2)
        self.setWindowFlags(QtCore.Qt.WindowMinimizeButtonHint | QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMaximizeButtonHint)
        self.dataGroupBox.setCurrentIndex(1)
        self.dataSelectionComboBox.activated['int'].connect(self.dataGroupBox.setCurrentIndex)
        self.dataSelectionComboBox.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(self)

    # Save File
    def file_save(self):
        name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', filter="*.mld", )
        if name[0] != '':
            with open(name[0], 'w') as file:
                text = self.dataPlainTextEdit.toPlainText()
                text2 = self.labelsPlainTextEdit.toPlainText()
                text3 = str(self.algorithmSelectionBox.currentText())
                algorithms = [LinearSVC(), KNeighborsClassifier(), DecisionTreeClassifier()]
                to_Write = text + ":" + text2 + ":" + text3
                letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']
                for i in range(len([j for j in algorithms[self.algorithmSelectionBox.currentIndex()].get_params()])):
                    to_Write += ":"
                    to_Write += eval("self.%s.text()" % letters[i])
                file.writelines(to_Write)

    # Open File
    def file_open(self):
        name = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', filter="*.mld")
        if name[0] != '':
            with open(name[0], 'r') as file:
                text = ('%s' % file.read())
            data, labels,algorithm= text.split(":")[:3]
            self.dataPlainTextEdit.setPlainText(data)
            self.labelsPlainTextEdit.setPlainText(labels)
            self.algorithmSelectionBox.setCurrentText(algorithm)

    # Add initial figure
    def addmpl(self, fig):
        self.canvas = FigureCanvas(fig)
        self.hbox3.addWidget(self.canvas, stretch=1)
        self.canvas.draw()
        self.toolbar = NavigationToolbar(self.canvas, self, coordinates=True)

    # Add 3d toolbar
    def addbar(self):
        self.toolbar = NavigationToolbar(self.canvas, self, coordinates=True)
        self.hbox3.addWidget(self.toolbar)

    # Remove 3d toolbar
    def rmbar(self):
        self.hbox3.removeWidget(self.toolbar)
        self.toolbar.close()

    # Number generator for labels and data
    def generateNumbers(self):
        text = eval(u'[%s]' % self.generateNumbersEdit.text())
        if len(text) == 5:
            dims, points, label, mini, maxi = text
            data = []
            labels = []
            label = [j for j in range(0, label + 1)]
            for i in range(1, points + 1):
                data_Point = []
                for j in range(dims):
                    data_Point.append(random.uniform(mini, maxi))
                data.append(data_Point)
                labels.append(random.randrange(min(label), max(label)))
            data = "%s" % data
            labels = "%s" % labels
            self.dataPlainTextEdit.setPlainText(data[1:-1])
            self.labelsPlainTextEdit.setPlainText(labels[1:-1])

    # Get and set parameters for chosen algorithm
    def algoParams(self, algorithm):
        y_pos = -22
        n = -1
        parameters = []
        letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']
        for i in sorted([j for j in algorithm.get_params()]):
            n += 1
            y_pos += 22
            statement = eval(u'algorithm.%s' % i)
            label = eval('self.%sl' % letters[n])
            textEdit = eval('self.%s' % letters[n])
            label.setGeometry(QtCore.QRect(0, y_pos, 100, 21))
            label.setObjectName("%sl" % letters[n])
            label.setText(str(i + ":"))
            textEdit.setGeometry(QtCore.QRect(105, y_pos, 170, 21))
            textEdit.setObjectName("%s" % letters[n])
            textEdit.setText(str(statement))
            self.update()
            parameters.append(statement)

    def next_Page(self):
        if self.current_page != self.max_Page:
            self.current_page += 1

    def previous_Page(self):
        if self.current_page != 0:
            self.current_page -= 1

    def closeEvent(self, event):
        close = QtWidgets.QMessageBox.question(self, 'Exit', "Are you sure you want to quit?",
                                               QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if close == QtWidgets.QMessageBox.Yes:
            event.accept()
            sys.exit()
        else:
            event.ignore()



