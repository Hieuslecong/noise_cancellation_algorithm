# Classifier
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import random
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV, ParameterGrid
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,recall_score,precision_score,fbeta_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import CategoricalNB
from sklearn.multioutput import ClassifierChain
from sklearn.naive_bayes import ComplementNB
from sklearn.tree._classes import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.tree._classes import ExtraTreeClassifier
from sklearn.ensemble._forest import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process._gpc import GaussianProcessClassifier
from sklearn.ensemble._gb import GradientBoostingClassifier
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingClassifier
from sklearn.neighbors._classification import KNeighborsClassifier
from sklearn.semi_supervised._label_propagation import LabelPropagation
from sklearn.semi_supervised._label_propagation import LabelSpreading
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm._classes import LinearSVC
from sklearn.linear_model._logistic import LogisticRegression
from sklearn.linear_model._logistic import LogisticRegressionCV
from sklearn.neural_network._multilayer_perceptron import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors._nearest_centroid import NearestCentroid
from sklearn.svm._classes import NuSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.linear_model._passive_aggressive import PassiveAggressiveClassifier
from sklearn.linear_model._perceptron import Perceptron
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors._classification import RadiusNeighborsClassifier
from sklearn.ensemble._forest import RandomForestClassifier
from sklearn.linear_model._ridge import RidgeClassifier
from sklearn.linear_model._ridge import RidgeClassifierCV
from sklearn.linear_model._stochastic_gradient import SGDClassifier
from sklearn.svm._classes import SVC
from sklearn.ensemble._stacking import StackingClassifier
from sklearn.ensemble._voting import VotingClassifier
import os
import argparse

import cv2
from skimage.measure import label,regionprops_table
import numpy as np 
import os
import glob
import pandas as pd
import pickle
import random
from skimage.feature import hog
import argparse
from skimage.morphology import skeletonize
from sklearn.metrics import f1_score,confusion_matrix,fbeta_score
import numpy as np

import cv2
from skimage.measure import label
import numpy as np 
import os
import glob
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import numpy as np
import cv2
import argparse
import glob
import pandas as pd
import datetime
import os
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse