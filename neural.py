# -*- coding: utf-8 -*-
import numpy as np
from neupy import algorithms, layers, plots
import pandas as pd
from sklearn import preprocessing
from sklearn.utils import shuffle


# noinspection PyTypeChecker
class Neural:
	input_features = 1
	output_classes = 1

	input_data = None
	output_data = None

	input_train = None
	output_train = None
	input_test = None
	output_test = None

	network = None
	layers = []
	labels = []
	activation_function_list = ('Linear', 'Sigmoid', 'HardSigmoid', 'Step',
								'Tanh', 'Relu', 'Softplus', 'Softmax', 'Elu', 'PRelu', 'LeakyRelu')

	df = None

	def __init__(self):
		pass

	def set_columns_split(self, inpu, output):
		self.input_features = inpu
		self.output_classes = output

	def isNumber(self, value):
		try:
			float(value)
			int(value)
			return True
		except ValueError:
			return False

	def read_file(self, path):
		try:
			self.df = pd.read_csv(path, header=None, skipinitialspace=True)
			self.df.fillna(method='bfill', inplace=True)
			self.df = shuffle(self.df)

			for idx, data_column in enumerate(self.df):
				contain_str = False  # TRUE - wszystkie kolumny beda  etykietami
				if not contain_str:
					for index in range(len(self.df[data_column])):
						val = self.df[data_column].iloc[index]
						if not self.isNumber(val):
							contain_str = True
							break
				if contain_str:
					unique_gat = self.df[data_column].unique()
					unique_gat_ind = []
					for ind in range(len(unique_gat)):
						unique_gat_ind.append(ind+1)
					x = dict(zip(unique_gat, unique_gat_ind))
					self.df[data_column] = self.df[data_column].map(x)
					self.labels.append(x)
					print('Etykiety dla kolumny: {0} {1}'.format(str(data_column + 1), str(x)))
				else:
					self.labels.append(None)
					print('Brak etykiety dla kolumny: {0}'.format(str(data_column + 1)))

			# ostatnia kolumna koniecznie jako etykieta
			last_col_index = max(self.df.columns)
			if self.labels[last_col_index] is None:
				self.labels.pop()
				unique_gat = self.df[last_col_index].unique()
				unique_gat_ind = []
				for ind in range(len(unique_gat)):
					unique_gat_ind.append(ind + 1)
				x = dict(zip(unique_gat, unique_gat_ind))
				self.df[last_col_index] = self.df[last_col_index].map(x)
				self.labels.append(x)
				print('Etykiety dla kolumny: {0} {1}'.format(str(last_col_index + 1), str(x)))

		except BaseException:
			raise IOError("Nie wczytano pliku")

	def prepare_data(self, verbose=False):
		if self.df is None:
			raise IOError("Nie załadowano pliku")

		self.input_data = np.array(self.df)
		self.input_data = np.delete(self.input_data,
									np.s_[self.input_features:self.input_features + 1],
									axis=1)  # usuwam kol wyjsciowe
		self.output_data = np.array(self.df)
		self.output_data = np.delete(self.output_data, np.s_[0:self.input_features], axis=1)  # usuwam dane wejsciowe

		# konwersja na klasy
		self.output_data = self.convert_to_classes(self.output_data)
		# self.input_data = self.normalize_data(self.input_data)

		if verbose:
			print("Dane wejściowe:\n" + str(self.input_data))
			print("Dane wyjściowe:\n" + str(self.output_data))

	def convert_to_classes(self, data):
		output_df = pd.DataFrame(data)
		unique_output = output_df[0].unique()
		empty_list = []

		for ind in range(len(unique_output)):
			empty_list.append(0)

		new_output = []
		for row in range(len(output_df)):
			for val in range(len(unique_output)):
				if output_df[0].values[row] == unique_output[val]:
					label = list(empty_list)
					label[val] = 1
					new_output.append(label)

		self.output_classes = len(unique_output)
		return np.array(new_output)


	def normalize_data(self, data, verbose=False):
		# normalizacja
		input_for_norm = pd.DataFrame(data)
		input_ = input_for_norm.values  # returns a numpy array
		min_max_scaler = preprocessing.MinMaxScaler()
		x_scaled = min_max_scaler.fit_transform(input_)
		out_ = pd.DataFrame(x_scaled)
		normalized = np.array(out_)

		if verbose:
			print(normalized)
		return normalized

	def split_data(self, percent):
		if self.df is None:
			raise IOError("Nie załadowano pliku")

		split_at = len(self.df) * (percent * 0.01)
		split_at = round(split_at, 0)
		print("Załadowano: " + str(len(self.df)) + ", Dane testowe od:" + str(int(split_at)))
		self.input_train = self.input_data[:int(split_at - 1)]
		self.output_train = self.output_data[:int(split_at - 1)]

		if percent != 100:
			self.input_test = self.input_data[int(split_at):]
			self.output_test = self.output_data[int(split_at):]

	def is_file_loaded(self):
		if self.df is None:
			return False
		if len(self.df) < 1:
			return False
		return True

	def select_algorithm(self, algorithm, options=None):
		try:
			self.network = algorithms.LevenbergMarquardt(self.layers)
			opt = options
			print(opt[1])
			print("Wybrano optymalizator: " + str(algorithm))
		except RecursionError:
			print("Problem rekursji")
			return None

		if algorithm == 'GradientDescent':
			self.network = algorithms.GradientDescent(self.layers)
		if algorithm == 'LevenbergMarquardt':
			self.network = algorithms.LevenbergMarquardt(connection=self.layers, mu=opt[0], mu_update_factor=opt[1])
		if algorithm == 'Adam':
			self.network = algorithms.Adam(self.layers)
		if algorithm == 'QuasiNewton':
			self.network = algorithms.QuasiNewton(self.layers)
		if algorithm == 'Quickprop':
			self.network = algorithms.Quickprop(self.layers)
		if algorithm == 'MinibatchGradientDescent':
			self.network = algorithms.MinibatchGradientDescent(self.layers)
		if algorithm == 'ConjugateGradient':
			self.network = algorithms.ConjugateGradient(self.layers)
		if algorithm == 'Hessian':
			self.network = algorithms.Hessian(self.layers)
		if algorithm == 'HessianDiagonal':
			self.network = algorithms.HessianDiagonal(self.layers)
		if algorithm == 'Momentum':
			self.network = algorithms.Momentum(self.layers)
		if algorithm == 'RPROP':
			self.network = algorithms.RPROP(self.layers)
		if algorithm == 'IRPROPPlus':
			self.network = algorithms.IRPROPPlus(self.layers)
		if algorithm == 'Adadelta':
			self.network = algorithms.Adadelta(self.layers)
		if algorithm == 'Adagrad':
			self.network = algorithms.Adagrad(self.layers)
		if algorithm == 'RMSProp':
			self.network = algorithms.RMSProp(self.layers)
		if algorithm == 'Adamax':
			self.network = algorithms.Adamax(self.layers)

	def decode_model(self, raw_model):
		try:
			decoded_model = []
			try:
				for layer in raw_model:
					single_layer = [layer[0], 'hidden', layer[1].get(), layer[2].get()]
					decoded_model.append(single_layer)
					# print("ID=" + str(layer[0]) + ", Neurons=" + str(layer[1].get()) + ", Activation=" + str(layer[2].get()))
				output_layer = decoded_model.pop()
			except Exception:
				return None
			output_layer[1] = 'output'
			output_layer[2] = self.output_classes
			decoded_model.append(output_layer)
			return decoded_model
		except Exception:
			return None

	def model_network(self, algorithm='LevenbergMarquardt', model=None, opt=None):

		model = self.decode_model(model)
		if model is None:
			model = [
				[1, 'hidden', 15, 'Linear'],
				[2, 'hidden', 10, 'Linear'],
				[3, 'output', self.output_classes, 'Elu']
			]
			# [Input(4), Elu(1)]
			# [Input(4), Elu(6), Elu(1)] EP: 100
		layer_model = [layers.Input(self.input_features)]
		for layer in model:
			if layer[3] == 'Linear':
				layer_model.append(layers.Linear(layer[2]))
			if layer[3] == 'Relu':
				layer_model.append(layers.Relu(layer[2]))
			if layer[3] == 'Sigmoid':
				layer_model.append(layers.Sigmoid(layer[2]))
			if layer[3] == 'HardSigmoid':
				layer_model.append(layers.HardSigmoid(layer[2]))
			if layer[3] == 'Step':
				layer_model.append(layers.Step(layer[2]))
			if layer[3] == 'Tanh':
				layer_model.append(layers.Tanh(layer[2]))
			if layer[3] == 'Softplus':
				layer_model.append(layers.Softplus(layer[2]))
			if layer[3] == 'Softmax':
				layer_model.append(layers.Softmax(layer[2]))
			if layer[3] == 'Elu':
				layer_model.append(layers.Elu(layer[2]))
			if layer[3] == 'PRelu':
				layer_model.append(layers.PRelu(layer[2]))
			if layer[3] == 'LeakyRelu':
				layer_model.append(layers.LeakyRelu(layer[2]))

		print('Model warstw: ' + str(layer_model))

		self.layers = layer_model
		self.select_algorithm(algorithm, options=opt)

	def show_plow(self):
		plots.error_plot(self.network)

	def train_network(self, verbose=False, shuffle=True, validate=True, epo=30):

		if self.network is None:
			print("Sieć nie została utworzona")
			raise ValueError("Sieć nie została utworzona")
		self.network.verbose = verbose
		self.network.shuffle_data = shuffle

		if self.input_train is None or self.output_train is None:
			print("Nie zainicjowano danych uczących")
			raise ValueError("Nie zainicjowano danych uczących")
		if validate:
			if self.input_test is None or self.output_test is None:
				print("Nie zainicjowano danych walidujących")
				raise ValueError("Nie zainicjowano danych walidujących")
			try:
				self.network.train(self.input_train, self.output_train, self.input_test, self.output_test, epochs=epo)
			except Exception:
				raise ValueError("Błąd w trakcie procesu uczenia!")
			#self.network.architecture()
		else:
			self.network.train(self.input_train, self.output_train, epochs=epo)

	def predict(self, data=None):
		try:
			y_class = self.network.predict(data)
			return y_class
		except AttributeError:
			return None

	def reset(self):
		self.network = None
		self.layers = None
		self.input_train = None
		self.output_train = None
		self.input_test = None
		self.input_train = None


if __name__ == '__main__':
	neural = Neural()

	neural.read_file('/Users/patryk/Dysk Google/ProjektSieciNeuronowe/iris.dat')
	# neural.read_file("C:\\Users\\patry\\Dysk Google\\ProjektSieciNeuronowe\\iris.dat")
	neural.set_columns_split(3, 2)
	neural.prepare_data(verbose=True)
	neural.split_data(80)
	neural.model_network('LevenbergMarquardt')
	neural.train_network(verbose=True, shuffle=True, validate=True, epo=10)
	neural.predict(None)
