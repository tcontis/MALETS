import UI,time, ast,warnings
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtWidgets
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

warnings.simplefilter("ignore", DeprecationWarning)

if __name__ == "__main__":
    #Launch app, show GUI
    app = QApplication([])
    form = UI.Ui_MainWindow()
    form.show()
    form.update()
    has_Bar = False
    page = 0
    tutorial = r'tutorial.txt'
    with open(tutorial, 'r',encoding='utf8') as file:
        tut_text = ('%s' % file.read())
    tut_text = [i for i in tut_text.split("@")]
    form.max_Page = len(tut_text) - 1

    #Set a figure
    fig = plt.figure(1, figsize=(1, 1),dpi=50)
    form.addmpl(fig)

    #Set variables for possible algorithms, and varibles of previous data
    algorithms = [LinearSVC(),KNeighborsClassifier(),DecisionTreeClassifier()]
    previous_X,previous_y,previous_X_length,previous_algo,previous_params,params,previous_pred = 0,0,0,0,[],[],0

    #Loop while app open
    while True:
        form.inputBox.setMaximumHeight(form.height() // 2)
        form.algorithmBox.setMaximumHeight(form.height() // 2)
        form.outputBox.setMaximumHeight(form.height() // 2)
        form.inputBox.setMaximumWidth(form.width() // 3)
        form.algorithmBox.setMaximumWidth(form.width() // 3)
        form.outputBox.setMaximumWidth(form.width() // 3)
        form.tutorialPlainTextEdit.setPlainText(tut_text[form.current_page])

        #Update events, make software less intensive by pausing it temporarily
        QtWidgets.QApplication.processEvents()
        time.sleep(0.01)

        #Try the code, errors may arise
        try:
            #Set the features, labels, and algorithms based on the corresponding textboxes
            X = ast.literal_eval((u'[%s]' % form.dataPlainTextEdit.toPlainText()))
            y = ast.literal_eval((u'[%s]' % form.labelsPlainTextEdit.toPlainText()))
            letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']
            algo = algorithms[form.algorithmSelectionBox.currentIndex()]

            #If algorithms are not the same, update the graph
            if previous_algo != algo:
                form.algoParams(algo)
                form.update()
                form.updateGeometry()
                previous_algo = algo
                QtWidgets.QApplication.processEvents()
            n = -1
            params=previous_params

            #Use user-entered parameters to update algorithms
            previous_params=[]

            for i in sorted([j for j in algo.get_params() if type(eval(u'algo.%s' % j))]):
                n += 1
                statement = eval(u'algo.%s' % i)
                types = type(statement)
                textEdit = eval('form.%s' % letters[n])
                exec(u'form.%sBox.show()' % letters[n])
                previous_params.append(str(textEdit.text()))
                if type(statement) is not type(None):
                     exec(u'algo.%s = types(textEdit.text())' % i)
            for j in letters[n+1:]:
                exec(u'form.%sBox.hide()' % j)

            #Check to make sure X and y are the same size.
            if len(y) > 1 and len(X) == len(y):

                #Split data into testing and training
                X_train, X_test, y_train, y_test = train_test_split(X, y)

                #Check to make sure training data includes at least 2 labels
                if len(set(y_train)) > 1:

                    #Fit algorithm, post scores
                    algo.fit(X_train, y_train)
                    if (y != previous_y) or (X != previous_X) or (len(X[0][:]) != previous_X_length) or previous_params != params:
                        form.outputPlainTextEdit.setPlainText("Accuracy Score: %s" % algo.score(X_test,y_test))

                    #Evaluate prediction slot if it contains a valid piece of data
                    try:
                        prediction_Text = form.outputPredictLineEdit.text()
                        if len(ast.literal_eval(u'%s' % prediction_Text)) == len(X[0][:]) and prediction_Text != previous_pred:
                            form.outputPlainTextEdit.appendPlainText(str("\nPrediction: %s" % algo.predict(ast.literal_eval(u'[%s]' % prediction_Text))))
                            previous_pred = prediction_Text
                    except SyntaxError:
                        pass
                    #If we are working with 2-dimensional data:
                    if len(X[0][:]) == 2:

                        #If the data has changed or the data has changed from 3d to 2d:
                        if (y != previous_y) or (X != previous_X) or (len(X[0][:]) != previous_X_length) or previous_params != params:

                            if has_Bar == False:
                                form.addbar()
                                has_Bar = True

                            #Remake graph
                            fig.clear()
                            fig.legends.clear()
                            plt.clf()
                            plt.cla()
                            plt.title("Features")
                            plt.xlabel("1st Feature")
                            plt.ylabel("2nd Feature")
                            plt.tight_layout()
                            markers = ['s','d','h','+','p','o','x','*','.',',','1','2','3','4','8']
                            n = -1
                            colors =['blue', 'red', 'white',"green","pink","orange","yellow","purple"]

                            # Show decision boundary
                            h = 0.2
                            X_2 = np.array(X)
                            x_min, x_max = X_2[:, 0].min() - 1, X_2[:, 0].max() + 1
                            y_min, y_max = X_2[:, 1].min() - 1, X_2[:, 1].max() + 1
                            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                                 np.arange(y_min, y_max, h))

                            # here "model" is your model's prediction (classification) function
                            Z = algo.predict(np.c_[xx.ravel(), yy.ravel()])

                            # Put the result into a color plot
                            Z = Z.reshape(xx.shape)
                            plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

                            #For every unique label:
                            for i in set(y):
                                n += 1

                                item1, item2 = [], []
                                #Get points where X data has the same position as the label, i, then scatter.
                                for num, item in enumerate(X):
                                    if any(num == a for a in [ynum for ynum, yitem in enumerate(y) if yitem == i]):
                                        item1.append(item[0])
                                        item2.append(item[1])

                                plt.scatter(x=item1,y=item2,marker=markers[n],c=colors[n],cmap=plt.cm.coolwarm)

                            #Plot legend with labels
                            fig.legend(handles=[mlines.Line2D([], [], color=colors[markers.index(i)], marker=i,
                              markersize=10, markevery=15)for i in markers[0:len(set(y))]], labels=[str(i) for i in set(y)])

                            #Set variables used to see if data changed
                            previous_y = y
                            previous_X = X
                            previous_X_length = len(X[0][:])
                            form.canvas.draw()

                    # If we are working with 3-dimensional data:
                    elif len(X[0][:]) == 3:

                        # If the data has changed or the data has changed from 2d to 3d:
                        if (y != previous_y) or (X != previous_X) or (len(X[0][:]) != previous_X_length) or previous_params != params:

                            # Remake graph
                            if has_Bar == False:
                                form.addbar()
                                has_Bar = True

                            fig.clear()
                            fig.legends.clear()
                            ax = Axes3D(fig, elev=-150, azim=110)
                            ax.set_title("Features")
                            ax.set_xlabel("1st Feature")
                            ax.set_ylabel("2nd Feature")
                            ax.set_zlabel("3rd Feature")

                            #Scatter points based on x,y,and z
                            ax.scatter([X[i][0] for i in range(len(X))], [X[i][1] for i in range(len(X))],
                                       [X[i][2] for i in range(len(X))], c=y)
                            fig.savefig('fig.png',bbox_inches='tight')
                            form.canvas.draw()
                            previous_y = y
                            previous_X = X
                            previous_X_length = len(X[0][:])

                    else:
                        if has_Bar == True:
                            form.rmbar()
                            has_bar = False

        #Catch exception, ignore it
        except Exception as e:
            pass

    sys.exit(app.exec_())
