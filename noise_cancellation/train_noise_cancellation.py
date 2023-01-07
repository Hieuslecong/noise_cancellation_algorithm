####################################################        
from .lib import *
from .function_cancellation import *
	####################################################
def train_model_cancellation(name_para,path_save_model,algos,seed,folds,path_parameter):
	# train_model_crack = argparse.ArgumentParser(description='Image Classification Using PyTorch', usage='[option] model_name')
	# train_model_crack.add_argument('--name_output', type=str, required=True)
	# args = train_model_crack.parse_args()
	if not os.path.exists(path_save_model):
		os.mkdir(path_save_model)
	
	if not os.path.exists(path_save_model+'/' +name_para):
		os.mkdir(path_save_model+'/' +name_para)
	kf = KFold(n_splits=folds,shuffle=True, random_state=seed)
	for k in range(folds):
		if name_para == 'EHR' :
			data = {
				"X_train": np.empty((1,3)),
				"X_test": np.empty((1,3)),
				"y_train": np.empty((1,)),
				"y_test": np.empty((1,))
			}
			para =[3,5,6]
		elif name_para == 'EH' :
			data = {
				"X_train": np.empty((1,2)),
				"X_test": np.empty((1,2)),
				"y_train": np.empty((1,)),
				"y_test": np.empty((1,))
			}
			para =[3,5]
		elif name_para == 'ER' :
			data = {
				"X_train": np.empty((1,2)),
				"X_test": np.empty((1,2)),
				"y_train": np.empty((1,)),
				"y_test": np.empty((1,))
			}
			para =[3,6]
		elif name_para == 'HR' :
			data = {
				"X_train": np.empty((1,2)),
				"X_test": np.empty((1,2)),
				"y_train": np.empty((1,)),
				"y_test": np.empty((1,))
			}
			para =[5,6]
		#"output_Data_EHR"
		for algo in algos:
			# Load dataset
			dataset = pd.read_excel('{}/{}_labeled.xlsx'.format(path_parameter,algo))
   			# Detemine feature cols
			X = dataset.iloc[:, para].values
			# Determine label col
			y = dataset.iloc[:, -1].values

			class0 = np.where(y==0)[0]
			class1 = np.where(y==1)[0]
			if len(class0) > len(class1):
				dataclass0 = X[class0[:len(class1)]]
				dataclass1 = X[class1]
				label0 = y[class0[:len(class1)]]
				label1 = y[class1]
			elif len(class0) < len(class1):
				dataclass1 = X[class1[:len(class0)]]
				dataclass0 = X[class0]
				label1 = y[class1[:len(class0)]]
				label0 = y[class0]
			else:
				dataclass1 = X[class1]
				dataclass0 = X[class0]
				label1 = y[class1]
				label0 = y[class0]

			# print(dataclass0.shape)
			# print(dataclass1.shape)
			# print(label0.shape)
			# print(label1.shape)
				
			data_dict = dict()
			for i,(train_index, test_index) in enumerate(kf.split(dataclass0)):
				X_train_sub,X_test_sub = np.vstack((dataclass0[train_index],dataclass1[train_index])), np.vstack((dataclass0[test_index],dataclass1[test_index]))
				y_train_sub,y_test_sub = np.hstack((label0[train_index],label1[train_index])), np.hstack((label0[test_index],label1[test_index]))

				data_dict["X_train_{}".format(i)] = np.array(X_train_sub)
				data_dict["X_test_{}".format(i)] = np.array(X_test_sub)
				data_dict["y_train_{}".format(i)] = np.array(y_train_sub)
				data_dict["y_test_{}".format(i)] = np.array(y_test_sub)
			data['X_train'] = np.vstack((data['X_train'],data_dict['X_train_{}'.format(k)]))
			data['X_test'] = np.vstack((data['X_test'],data_dict['X_test_{}'.format(k)]))
			data['y_train'] = np.hstack((data['y_train'],data_dict['y_train_{}'.format(k)]))
			data['y_test'] = np.hstack((data['y_test'],data_dict['y_test_{}'.format(k)]))

		data['X_train'] = np.delete(data['X_train'],0,0)
		data['X_test'] = np.delete(data['X_test'],0,0)
		data['y_train'] = np.delete(data['y_train'],0,0)
		data['y_test'] = np.delete(data['y_test'],0,0)

		print(data["X_train"].shape)
		print(data["X_test"].shape)
		print(data["y_train"].shape)
		print(data["y_test"].shape)

		sc = MinMaxScaler()
		X_train = data["X_train"]
		X_test = data["X_test"]
		X_train = sc.fit_transform(X_train)
		X_test = sc.transform(X_test)
		y_train = data["y_train"]
		y_test = data["y_test"]


		parameters = {
			"n_estimators": list(range(20,121)),
			"max_depth":[1,2,3,4],
			"learning_rate":list(np.array(range(40,61))*0.01)
		}
		para = list(ParameterGrid(parameters))
		f= []; r = []; p =[]; a = []

		# for k in para:
		"""
		Choose A Classifier
		"""
		classifier = GradientBoostingClassifier()

		"""
		Training
		"""
		classifier.fit(X_train, y_train)

		# Predicting the Test set results
		y_pred = classifier.predict(X_test)

		# Making the Confusion Matrix

		cm = confusion_matrix(y_test, y_pred)
		ac = accuracy_score(y_test, y_pred)
		f1 = fbeta_score(y_test, y_pred,beta=1)
		p = precision_score(y_test, y_pred)
		r = recall_score(y_test, y_pred)

		print("Confusion Matrix: ",cm)
		print("Accuracy: ",ac)
		print("F1-score: ",f1)
		print("Precision: ",p)
		print("Recall: ",r)

		scores = cross_val_score(classifier, X_train, y_train, cv=5)
		print("%0.4f accuracy with a standard deviation of %0.4f" % (scores.mean(), scores.std()))

	# from sklearn.model_selection import GridSearchCV,RandomizedSearchCV, ParameterGrid
	# parameters = {
	#     "n_estimators": list(range(20,151)),
	#     "max_depth":[1,2,3,4],
	#     "learning_rate":list(np.array(range(1,200))*0.01)
	# }
	# para = ParameterGrid(parameters)
	# cv = RandomizedSearchCV(classifier,parameters,cv=10,scoring='f1')
	# cv.fit(X_train,y_train)

	# def display(results):
	#     print(f'Best parameters are: {results.best_params_}')
	#     print("\n")
	#     mean_score = results.cv_results_['mean_test_score']
	#     std_score = results.cv_results_['std_test_score']
	#     params = results.cv_results_['params']
	#     for mean,std,params in zip(mean_score,std_score,params):
	#         print(f'{round(mean,3)} + or -{round(std,3)} for the {params}')
			
	# display(cv)
		"""
		Save model and scale
		"""
		# pickle.dump(classifier,open('/content/drive/MyDrive/classifierforcrack/models/Model_balance{}.model'.format(k),'wb'))
		# pickle.dump(sc,open('/content/drive/MyDrive/classifierforcrack/models/Normalization_balance{}.pickle'.format(k),'wb'))
		f = open('{}/{}/Normalization_balance{}'.format(path_save_model,name_para,k),'wb')
		f.write(pickle.dumps(sc))
		f.close()
		f = open('{}/{}/Model_balance{}'.format(path_save_model,name_para,k),'wb')
		f.write(pickle.dumps(classifier))
		f.close()

if __name__ == '__main__': 
    name_para='EHR'
    path_save_model = r'D:\pix2pixHD\code\noise_cancellation_algorithm\output\model_cancellation/model_EHR_{}'.format(name_para)
    seed = 42
    folds = 2
    algos = ["Deepcrack","EHCNN","FCN","HRNBM0.3","HED","Unet"]
    path_parameter='D:/pix2pixHD/code/noise_cancellation_algorithm/output/image_area'
    train_model_cancellation(name_para,path_save_model,algos,seed,folds,path_parameter)