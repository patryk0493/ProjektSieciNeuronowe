# -*- coding: utf-8 -*-
import tkinter.messagebox as msg_box
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from neural import Neural
from VerticalScrolledFrame import VerticalScrolledFrame
import numpy as np
import csv


class GUI:
	top = Tk()
	selected_file_path = NONE
	select_file_label_textvariable = IntVar()
	splitdata_spinbox_textvariable = IntVar()
	optimalizers_str_var = StringVar()
	padding_val = 7
	input_columns = IntVar()
	output_columns = IntVar()
	verbose = BooleanVar()
	shuffle_data = BooleanVar()
	epochs = IntVar()
	validate = BooleanVar()
	plot = BooleanVar()
	mu = StringVar()
	mu_update = StringVar()
	test_string = StringVar()
	file_info = StringVar()
	optimalizers = ['LevenbergMarquardt', 'GradientDescent', 'Adam', 'QuasiNewton', 'Quickprop',
					'MinibatchGradientDescent', 'ConjugateGradient', 'Hessian', 'HessianDiagonal',
					'Momentum', 'RPROP', 'IRPROPPlus', 'Adadelta', 'Adagrad', 'RMSProp', 'Adamax']
	activation_function_list = ('Linear', 'Sigmoid', 'HardSigmoid', 'Step',
		'Tanh', 'Relu', 'Softplus', 'Softmax', 'Elu', 'PRelu', 'LeakyRelu')

	layers_count = 0
	layers_continous_count = 0
	raw_model = []

	network_frame = None
	layers_frame = None
	optimalizers_combobox = None
	optimalizer_opt_frame = None

	neural = Neural()

	def __init__(self):

		self.test_entry = None
		self.select_file_label_textvariable.set("Nie wybrano pliku")
		self.selected_optimalizer = self.optimalizers[0]

		self.input_columns.set(4)
		self.output_columns.set(1)

		self.verbose.set(1)
		self.shuffle_data.set(1)
		self.epochs.set(0)
		self.validate.set(1)
		self.plot.set(0)
		self.test_string.set('')
		self.mu.set(0.1)
		self.mu_update.set(1.2)
		self.file_info.set('Brak informacji o pliku')

	def add_LayerCallBack(self):

		layer_frame = LabelFrame(self.layers_frame.interior, relief=FLAT)
		layer_frame.pack(fill="both", expand="no", side=TOP)

		Label(layer_frame, text='Ilość neuronów: ').pack(side=LEFT)

		neuron_count_var = IntVar()
		neuron_count_spinbox = Spinbox(layer_frame, from_=1, to=10000, width=6,
										textvariable=neuron_count_var)
		neuron_count_spinbox.pack(side=LEFT, pady=self.padding_val, padx=self.padding_val)
		neuron_count_var.set(self.output_columns.get())

		activation_fuction = StringVar()
		activation_function_combobox = ttk.Combobox(layer_frame,
				values=self.activation_function_list,
				textvariable=activation_fuction,
				state="readonly", width=12)
		activation_fuction.set(self.activation_function_list[0])
		activation_function_combobox.pack(pady=self.padding_val, padx=self.padding_val, side=LEFT)

		index = self.layers_continous_count
		Button(layer_frame, text="Usuń warstwę", command=lambda: remove_layer(index)).pack(side=LEFT,
														pady=self.padding_val, padx=self.padding_val)

		neuron_count_var.set(self.validate_val(self.output_columns.get(), 1, 10000, neuron_count_var)[0])
		layer_data = [self.layers_continous_count, neuron_count_var, activation_fuction]

		self.raw_model.append(layer_data)
		self.layers_continous_count += 1
		self.layers_count += 1

		def remove_layer(layer_id):
			ind = 0
			for layer in self.raw_model:
				if int(layer[0]) == int(layer_id):
					print("USUWAM index:{0} ID={1}, Neurons={2}, Activation={3}".format(str(ind), str(layer[0]),
							str(layer[1]), str(layer[2].get())))
					self.raw_model.pop(ind)
					layer_frame.destroy()
					self.layers_count -= 1
					break
				else:
					ind += 1

		print('Warstwy: ' + str(self.layers_count))

	def selectFileCallback(self):
		self.selected_file_path = filedialog.askopenfilename(title='Wybierz plik z danymi',
															filetypes=[("All files", "*.dat"), ("All files", "*.csv")])
		if self.selected_file_path:
			msg_box.showinfo("Informacja", "Otwarto plik: " + self.selected_file_path)
			self.neural.labels = []
			file_name_tokens = self.selected_file_path.split("/")
			rel_file_name = file_name_tokens[len(file_name_tokens) - 1]
			self.select_file_label_textvariable.set("Wybrano: " + rel_file_name)
			print("Wybrano plik: " + rel_file_name)
			self.neural.read_file(path=self.selected_file_path)
			self.file_info.set(
				'Załadowano rekordów: ' + str(len(self.neural.df)) + ', kolumn: ' + str(len(self.neural.df.columns)))
			self.input_columns.set(len(self.neural.df.columns) - 1)

	def optimalizerCallback(self):
		self.selected_optimalizer = self.optimalizers_combobox.get()
		if self.selected_optimalizer == 'LevenbergMarquardt':
			self.optimalizer_opt_frame.pack(side=TOP, pady=self.padding_val, padx=self.padding_val)
		else:
			self.optimalizer_opt_frame.pack_forget()
		print("Optymalizer: " + self.selected_optimalizer)

	def reset_LayerCallBack(self):
		self.layers_count = 0
		for widget in self.layers_frame.interior.winfo_children():
			widget.destroy()
		self.raw_model.clear()
		msg_box.showinfo("Informacja", "Zresetowano model sieci")

	def startLearnCallback(self):

		if self.neural.is_file_loaded():
			result = self.validate_val(default=80, min_=1, max_=100, var=self.splitdata_spinbox_textvariable)
			if result[1]:  # invalid value
				msg_box.showinfo("Informacja", "Niewłaściwa wartość podziału danych, ustawiam: " + str(result[0]))
				self.epochs.set(result[0])

			result = self.validate_val(default=10, min_=1, max_=100000, var=self.epochs)
			if result[1]:  # invalid value
				msg_box.showinfo("Informacja", "Niewłaściwa wartość liczby epok, ustawiam: " + str(result[0]))
				self.epochs.set(result[0])

			result = self.is_digit(self.mu.get())
			if not result:
				msg_box.showinfo("Informacja", "Niewłaściwa wartość współczynnika µ, przerywam i ustawiam: 0.1")
				self.mu.set("0.1")
				return

			result = self.is_digit(self.mu_update.get())
			if not result:
				msg_box.showinfo("Informacja", "Niewłaściwa wartość współczynnika β, przerywam i  ustawiam: 1.2")
				self.mu_update.set("1.2")
				return

			self.neural.set_columns_split(self.input_columns.get(), self.output_columns.get())
			print(str("Inputs: " + str(self.input_columns.get()) + ", Outputs: " + str(self.output_columns.get())))

			try:
				self.neural.prepare_data(verbose=False)
			except IOError:
				msg_box.showerror("Informacja", "Nie wybrano pliku z danymi")
				return

			self.neural.split_data(self.splitdata_spinbox_textvariable.get())

			try:
				options_ = [float(self.mu.get()), float(self.mu_update.get())]
				self.neural.model_network(algorithm=self.selected_optimalizer, model=self.raw_model, opt=options_)
			except ValueError:
				msg_box.showerror(title="Błąd", message="Nie udało się nauczyć sieci!")
				return
			try:
				self.neural.train_network(verbose=True, shuffle=self.shuffle_data.get(),
											validate=self.validate.get(),
											epo=self.epochs.get())
			except ValueError as e:
				msg_box.showerror("Błąd uczenia sieci", "Przerwano uczenie sieci" + str(e))
				return

			if self.plot.get():
				try:
					self.neural.show_plow()
				except AttributeError:
					msg_box.showerror("Błąd", "Nie udało się zaprezentowac wykresu precesu uczenia ")

			if self.verbose.get():
				top = Toplevel()

				def callback():
					top.destroy()

				close_window = Button(top, text="Zamknij okno", command=callback)
				close_window.pack(side=BOTTOM, pady=self.padding_val, padx=self.padding_val)

				top.protocol("WM_DELETE_WINDOW", callback)
				var = StringVar()
				Label(top, textvariable=var, relief=FLAT).pack(side=TOP)
				var.set("Wynik uczenia sieci:")

				text = Text(top)
				text.pack(side=TOP, pady=self.padding_val, padx=self.padding_val, fill="both", expand="no")

				text.insert(END, 'Plik: ' + str(self.selected_file_path) + '\n')
				text.insert(END, 'Załadowano rekordów: ' + str(len(self.neural.df)) + ', kolumn: ' + str(
					len(self.neural.df.columns)) + '\n')
				text.insert(END, str("Wejścia: " + str(self.input_columns.get()) + ", Wyjścia: " + str(
					self.output_columns.get())) + '\n')
				text.insert(END, 'Model warstw: ' + str(self.neural.layers) + '\n')
				text.insert(END, 'Liczba epok: ' + str(self.epochs.get()) + '\n')
				text.insert(END, 'Algorytm: ' + str(self.selected_optimalizer) + '\n\n')
				text.insert(END, 'Proces uczenia:\n')
				try:
					for epoch in range(self.neural.network.last_epoch):
						if self.validate.get():
							text.insert(END, 'Epoka:{:>5}, błąd uczenia:{:>11}, błąd walidacji: {:>11} \n'.format(epoch + 1,
								"%.5f" % self.neural.network.errors[epoch], "%.5f" % self.neural.network.validation_errors[epoch]))
						else:
							text.insert(END, 'Epoka {:>5}, błąd uczenia: {:>11},\n'.format(epoch + 1,
												"%.5f" % self.neural.network.errors[epoch]))
				except Exception:
					msg_box.showerror("Błąd", "Błąd przy logowaniu danych uczenia sieci")

				top.minsize(500, 400)
				top.title = 'Wyniki uczenia sieci'
				top.mainloop()
			else:
				msg_box.showinfo("Informacja", "Zakończono uczenie sieci !")

			print("Nauczono sieć")
		else:
			msg_box.showinfo("Informacja", "Nie załadowano pliku!")

	def startClassificationCallback(self):
		if self.neural.network is not None:
			data = self.test_string.get()
			data_list = []
			predicted_val = None
			test_val = self.neural.output_test
			stacked = []
			if data != '':
				print("a: " + data)
				try:
					data_list = self.csv_line_to_list(data)
				except IndexError:
					msg_box.showerror("Błąd dekompozycji danych", "Niepoprawne dane")
					return
				print("b: " + str(data_list))
				input_selected = np.array([data_list])
				# input_selected = data_list
			else:
				input_selected = self.neural.input_test

			#print(input_selected)

			try:
				predicted_val = self.neural.predict(data=input_selected)
				# stacked = np.hstack((test_val, np.asarray(predicted_val)))
			except Exception:
				msg_box.showerror("Błąd podczas klasyfikacji danych", "Niepoprawne dane wejściowe")
				return

			print("Zakończono klasyfikację")

			top = Toplevel()

			def callback():
				top.destroy()

			close_window = Button(top, text="Zamknij okno", command=callback)
			close_window.pack(side=BOTTOM, pady=self.padding_val, padx=self.padding_val)

			top.protocol("WM_DELETE_WINDOW", callback)
			var = StringVar()
			Label(top, textvariable=var, relief=FLAT).pack(side=TOP)

			text = Text(top)
			text.pack(side=TOP, pady=self.padding_val, padx=self.padding_val, fill="both", expand="yes")

			text.tag_configure('good', background='green')
			text.tag_configure('bad', background='red')

			labels = self.neural.labels

			for lab in labels[-1]:
				text.insert(END, ' {:>12},'.format(str(lab)))

			text.insert(END, '  Klasa wyjściowa\n')
			element_index = 0
			well_classed = 0

			for element in predicted_val:
				var.set("Wynik klasyfikacji dla danych testowych")
				label_predicted = 'Błąd'
				label_real = 'Błąd'
				if data == '':
					test_hightest_index = 0
					predicted_hightest_index = 0

					for i in range(len(test_val[element_index])):
						if test_val[element_index][i] > test_hightest_index:
							test_hightest_index = i

					for index in range(len(predicted_val[element_index])):
						if predicted_val[element_index][index] >= predicted_val[element_index][predicted_hightest_index]:
							predicted_hightest_index = index

					for name, dict_ in labels[-1].items():
						if predicted_hightest_index + 1 == dict_:
							label_predicted = name

					for name, dict_ in labels[-1].items():
						if test_hightest_index + 1 == dict_:
							label_real = name

					color = 'bad'
					if predicted_hightest_index == test_hightest_index:
						color = 'good'
						well_classed += 1

					for list_element in range(len(element)):
						text.insert(END, ' {:>12},'.format("%.3f" % element[list_element]), color)

					text.insert(END, ' Sklasyfikowano jako: {:>12}, wartość rzecz. {:>12} \n'
													.format(str(label_predicted), str(label_real)), color)

					element_index += 1
				else:
					var.set("Wynik klasyfikacji dla zadanej próbki" + str(predicted_val[0]))
					print(predicted_val[0])

					for list_element in range(len(predicted_val[0])):
						print("%.3f" % predicted_val[0][list_element])
						text.insert(END, ' {:>12},'.format("%.5f" % predicted_val[0][list_element]))

					predicted_hightest_index = 0
					for index in range(len(predicted_val[0])):
						if predicted_val[0][index] >= predicted_val[0][predicted_hightest_index]:
							predicted_hightest_index = index

					for name, dict_ in labels[-1].items():
						if predicted_hightest_index + 1 == dict_:
							label_predicted = name

					text.insert(END, ' Sklasyfikowano jako: {:>12}\n'.format(str(label_predicted)))

			if len(predicted_val) > 1:
				percent = (well_classed / len(predicted_val)) * 100
				Label(top, text="Poprawnie klasyfikowano {} ({}%) z {} ".format(str(well_classed), str("%.2f" % percent), str(len(predicted_val))),
						relief=FLAT).pack(side=RIGHT)

			top.minsize(800, 600)
			top.title = 'Wyniki klasyfikacji'
			top.mainloop()

		else:
			msg_box.showerror("Błąd", "Nie nauczono sieci!")

	def csv_line_to_list(self, line):
		try:
			from StringIO import StringIO
		except ImportError:
			from io import StringIO
		output = []
		f = StringIO(line)
		reader = csv.reader(f, delimiter=',', skipinitialspace=True)

		for idy, row in enumerate(reader):
			labels = self.neural.labels
			converted = []
			for idx, val in enumerate(row):
				if labels[idx] is not None:
					for name, dict_ in labels[idx].items():
						if str(name) == val:
							converted.append(dict_)
							break
				else:
					converted.append(val)
			output = list(converted)
		return output

	def update_input_spinbox(self):
		if self.neural.is_file_loaded():
			if self.input_columns.get() == len(self.neural.df.columns):
				self.input_columns.set(len(self.neural.df.columns) - 1)

	def is_digit(self, x):
		try:
			float(x)
			return True
		except ValueError:
			return False

	def validate_val(self, default, min_, max_, var):
		# return false if ALL OKAY
		try:
			var = var.get()
			if self.is_digit(var):
				if (var < min_) or (var > max_):
					print('Wartość poza zakresem')
					var.set(default)
					return [var, True]
				else:
					print('Wartość jest okej')
					return [var, False]
			else:
				print('Wartość nie jest cyfrą, zwracam: ' + default.__str__())
				var.set(default)
				return [var, True]
		except Exception:
			try:
				var = default
				print('Wyjątek niewłaściwa wartość, zwracam: ' + default.__str__())
				var.set(default)
				return [var, True]
			except AttributeError:
				print('Wyjątek niewłaściwa wartość, zwracam: ' + var.__str__())
				return [var, True]


	def createGUI(self):

		data_frame = LabelFrame(self.top, text="Wybór danych")
		data_frame.pack(fill="both", expand="no", pady=self.padding_val, padx=self.padding_val)

		select_file_btn = Button(data_frame, text="Wybierz plik", command=self.selectFileCallback)
		select_file_btn.pack(side=LEFT, pady=self.padding_val, padx=self.padding_val)
		select_file_label = Label(data_frame, textvariable=self.select_file_label_textvariable, relief=FLAT).pack(
			side=LEFT, pady=self.padding_val, padx=self.padding_val)
		splitdata_spinbox = Spinbox(data_frame, from_=0, to=100, width=4, textvariable=self.splitdata_spinbox_textvariable)
		splitdata_spinbox.pack(side=RIGHT, pady=self.padding_val, padx=self.padding_val)
		self.splitdata_spinbox_textvariable.set("80")
		Label(data_frame, text='Procent danych uczących', relief=FLAT).pack(side=RIGHT)

		Label(data_frame, text="Ilość wejść", relief=FLAT).pack(side=BOTTOM)
		input_columns_spinbox = Spinbox(data_frame, from_=1, to=10000, width=10, textvariable=self.input_columns,
										state='readonly', command=self.update_input_spinbox)
		input_columns_spinbox.pack(side=BOTTOM, pady=self.padding_val, padx=self.padding_val)
		Label(data_frame, textvariable=self.file_info, relief=FLAT).pack(side=TOP)

		# === MODEL FRAME ===
		model_frame = LabelFrame(self.top, text="Modelowanie sieci")
		model_frame.pack(fill="both", expand="no", pady=self.padding_val, padx=self.padding_val)

		optimalizer_frame = LabelFrame(model_frame, text="Wybór optymalizatora")
		optimalizer_frame.pack(expand="no", fill='y', pady=self.padding_val, padx=self.padding_val, side=LEFT)
		# optimalizer_frame.place(relwidth = 0.5)
		self.optimalizers_combobox = ttk.Combobox(optimalizer_frame, values=self.optimalizers,
													textvariable=self.optimalizers_str_var,
													state="readonly")
		self.optimalizers_str_var.set(self.optimalizers[0])
		self.optimalizers_combobox.pack(pady=self.padding_val, padx=self.padding_val)
		self.optimalizers_combobox.bind("<<ComboboxSelected>>", lambda x: self.optimalizerCallback())

		self.optimalizer_opt_frame = LabelFrame(optimalizer_frame)
		self.optimalizer_opt_frame.pack(expand="yes", fill='both', pady=self.padding_val, padx=self.padding_val, side=LEFT)

		Label(self.optimalizer_opt_frame, text="Czynnik - µ", relief=FLAT).pack(side=TOP)
		mu_entry = ttk.Entry(self.optimalizer_opt_frame, textvariable=self.mu)
		mu_entry.pack(side=TOP, pady=self.padding_val, padx=self.padding_val)

		Label(self.optimalizer_opt_frame, text="Czynnik regularyzacyjny - β", relief=FLAT).pack(side=TOP)
		mu_update_entry = ttk.Entry(self.optimalizer_opt_frame, textvariable=self.mu_update)
		mu_update_entry.pack(side=TOP, pady=self.padding_val, padx=self.padding_val)

		options_frame = LabelFrame(model_frame, text="Opcje uczenia")
		options_frame.pack(expand="no", fill='y', pady=self.padding_val, padx=self.padding_val, side=LEFT)
		epochs_label = Label(options_frame, text="Liczba epok:", relief=FLAT).pack()
		self.epochs.set(10)
		Spinbox(options_frame, from_=1, to=10000, width=6, textvariable=self.epochs).pack(
			pady=self.padding_val,
			padx=self.padding_val)
		shuffle_data_checkbox = Checkbutton(options_frame, text="Mieszanie danych", variable=self.shuffle_data,
											onvalue=1, offvalue=0)
		shuffle_data_checkbox.pack(pady=self.padding_val, padx=self.padding_val)
		Checkbutton(options_frame, text="Pokaż szczegóły", variable=self.verbose, onvalue=1,
										offvalue=0).pack(pady=self.padding_val, padx=self.padding_val)

		# === Network FRAME ===
		self.network_frame = LabelFrame(model_frame, text="Sieć")
		self.network_frame.pack(fill="both", expand="no", pady=self.padding_val, padx=self.padding_val, side=TOP)
		self.layers_frame = VerticalScrolledFrame(self.network_frame, relief=FLAT)  # warstwy
		self.layers_frame.pack(fill="both", expand="no", side=TOP)

		reset_layer_button = Button(self.network_frame, text="Resetuj sieć", command=self.reset_LayerCallBack)
		reset_layer_button.pack(side=RIGHT, pady=self.padding_val, padx=self.padding_val)
		add_layer_button = Button(self.network_frame, text="Dodaj warstwę sieci", command=self.add_LayerCallBack)
		add_layer_button.pack(side=RIGHT, pady=self.padding_val, padx=self.padding_val)

		# === Simulation FRAME ===
		simulation_frame = LabelFrame(self.top, text="Uczenie sieci")
		simulation_frame.pack(fill="both", expand="no", pady=self.padding_val, padx=self.padding_val)

		Checkbutton(simulation_frame, text="Ucz. z danymi walidującymi", variable=self.validate,
										onvalue=1, offvalue=0).pack(pady=self.padding_val, padx=self.padding_val,
										side=LEFT)

		Checkbutton(simulation_frame, text="Pokaż wykres [OSTROŻNIE!!!]", variable=self.plot, onvalue=1,
									offvalue=0).pack(pady=self.padding_val, padx=self.padding_val, side=LEFT)

		start_learn_button = Button(simulation_frame, text="Rozpocznij uczenie sieci", command=self.startLearnCallback)
		start_learn_button.pack(side=RIGHT, pady=self.padding_val, padx=self.padding_val)

		# === test FRAME ===
		test_frame = LabelFrame(self.top, text="Testowanie sieci")
		test_frame.pack(fill="both", expand="no", pady=self.padding_val, padx=self.padding_val)

		# L1 = Label(test_frame, text="hjkhjk").pack()

		Label(test_frame, text="Podaj dane do klasyfikacji", relief=FLAT).pack(side=LEFT)

		self.test_entry = ttk.Entry(test_frame, textvariable=self.test_string).pack(side=LEFT)
		start_test_button = Button(test_frame, text="Rozpocznij klasyfikację", command=self.startClassificationCallback)
		start_test_button.pack(side=LEFT, pady=self.padding_val, padx=self.padding_val)

	def start_gui(self):
		self.add_LayerCallBack()
		self.top.title("Klasyfikator")
		self.top.minsize(950, 600)
		self.top.maxsize(800, 400)
		self.top.mainloop()


gui = GUI()
gui.createGUI()
gui.start_gui()
