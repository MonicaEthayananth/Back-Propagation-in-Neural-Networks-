# Ethayananth, Monica Rani
# 1001-417-942
#2016-11-15

import numpy as np
import Tkinter as Tk
import matplotlib
import os
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pdb
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import colorsys
import scipy.misc
from PIL import Image
import random
from sklearn.preprocessing import normalize
import numpy as np
import theano
from sklearn.utils import shuffle
import theano.tensor as T
from sklearn import metrics
from theano.d3viz import d3viz

FOLDER = "C:\\Users\\Monica\\Documents\\neural networks\\assign 4\\train" #folder containing the images
TEST_FOLDER = "C:\\Users\\Monica\\Documents\\neural networks\\assign 4\\test"  # folder containing the images

import pdb



class ClDataSet:
    # This class encapsulates the data set
    # The data set includes input samples and targets
    def __init__(self, samples=None, targets=None):
        for s, p, f in os.walk(FOLDER):
            #pdb.set_trace()
            Images = []
            Labels = []
            np.random.shuffle(f)
            for imgs in f:
                path = os.path.join(s, imgs)
                image = Image.open(path)
                image = np.array(image,dtype=np.float64)
                image /= 255.0
                image = image.reshape(-1)
                Images.append(image)

                # Labels.append(imgs.split("_")[0])
                #Labels.append(int(imgs.split("_")[0]))
                label = np.zeros(10, dtype=np.float64)
                label[int(imgs.split("_")[0])] = 1.0
                Labels.append(label)
        self.inputs = np.array(Images)
        self.targets = np.array(Labels)

        #self.targets = np.array(Labels,dtype=np.float64).reshape(-1,1)
        #print self.samples.shape,self.desired_target_vectors.shape
        self.inputs,self.targets = shuffle(self.inputs,self.targets)

nn_experiment_default_settings = {
    # Optional settings
    "min_initial_weights": -1.0,  # minimum initial weight
    "max_initial_weights": 1.0,  # maximum initial weight
    "number_of_inputs": 785,  # number of inputs to the network
    "train_network": 0.00001,  # learning rate
    "delayed_elements":1, #number of delayed elements
    "hidden_nodes":500,#select the nodes in the hidden layer
    "number_iterations":1,# number of times system goes over the samples.
    "momentum": 0.1,  # momentum
    "weight_regularization": 0.5,  # lamda value
   # "layers_specification": [{"number_of_neurons": 10, "activation_function": "linear"}],  # list of dictionaries
    'number_of_classes': 10,
    'number_of_samples_in_each_class': 3
}






class ClNNGui2d:
    """
    This class presents an experiment to demonstrate
    Perceptron learning in 2d space.
    Farhad Kamangar 2016_09_02
    """

    def __init__(self, master):
        self.master = master
        self.xmin = 0
        self.xmax = 200
        self.ymin = 0
        self.ymax = 1
        self.master.update()
        self.dataset = ClDataSet()
        self.learning_method_variable ="Relu"
        self.step_size = 1
        self.current_sample_loss = 0
        self.sample_points = []
        self.target = []
        self.sample_colors = []
        self.weights = np.array([])
        self.class_ids = np.array([])
        self.output = np.array([])
        self.desired_target_vectors = np.array([])
        self.xx = np.array([])
        self.yy = np.array([])
        self.loss_type = ""
        self.master.rowconfigure(0, weight=2, uniform="group1")
        self.master.rowconfigure(1, weight=1, uniform="group1")
        self.master.columnconfigure(0, weight=2, uniform="group1")
        self.master.columnconfigure(1, weight=1, uniform="group1")

        self.canvas = Tk.Canvas(self.master)
        self.display_frame = Tk.Frame(self.master)
        self.display_frame.grid(row=0, column=0, columnspan=2, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.display_frame.rowconfigure(0, weight=1)
        self.display_frame.columnconfigure(0, weight=1)
        self.figure = plt.figure("Back Propagation")
        self.axes = self.figure.add_subplot(111)
        self.figure = plt.figure("Back Propagation")
        self.axes = self.figure.add_subplot(111)
        plt.title("Back Propagation")

        plt.xlim(self.xmin,self.xmax)
        plt.ylim(self.ymin,self.ymax)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.display_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        # Create sliders frame
        self.sliders_frame = Tk.Frame(self.master)
        self.sliders_frame.grid(row=1, column=0)
        self.sliders_frame.rowconfigure(0, weight=10)
        self.sliders_frame.rowconfigure(1, weight=2)
        self.sliders_frame.columnconfigure(0, weight=1, uniform='s1')
        self.sliders_frame.columnconfigure(1, weight=1, uniform='s1')
        # Create buttons frame
        self.buttons_frame = Tk.Frame(self.master)
        self.buttons_frame.grid(row=1, column=1)
        self.buttons_frame.rowconfigure(0, weight=1)
        self.buttons_frame.columnconfigure(0, weight=1, uniform='b1')
        # Set up the sliders
        ivar = Tk.IntVar()


# slider for training network
        self.train_network_slider_label = Tk.Label(self.sliders_frame, text="Train Nework")
        self.train_network_slider_label.grid(row=1, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.train_network_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                                from_=0.00001, to_=1, resolution=0.00001, bg="#DDDDDD",
                                                activebackground="#FF0000",
                                                highlightcolor="#00FFFF", width=10,
                                                command=lambda event: self.train_network_slider_callback())
        self.train_network_slider.set(0.00001)
        self.alpha = self.train_network_slider.get()
        self.train_network_slider.bind("<ButtonRelease-1>", lambda event: self.train_network_slider_callback())
        self.train_network_slider.grid(row=1, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
# slider for hidden layer
        self.hidden_layer_slider_label = Tk.Label(self.sliders_frame, text="nodes in hidden layer")
        self.hidden_layer_slider_label.grid(row=2, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.hidden_layer_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                             from_=1, to_=500, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF", width=10,
                                             command=lambda event: self.hidden_layer_slider_callback())
        self.hidden_layer_slider.set(100)
        self.hidden_layer = self.hidden_layer_slider.get()
        self.hidden_layer_slider.bind("<ButtonRelease-1>", lambda event: self.hidden_layer_slider_callback())
        self.hidden_layer_slider.grid(row=2, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        #slider for lamda
        self.weight_regularization_slider_label = Tk.Label(self.sliders_frame, text="lambda")
        self.weight_regularization_slider_label.grid(row=3, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.weight_regularization_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                             from_=0.0, to_=0.5,resolution=0.1, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF", width=10,
                                             command=lambda event: self.weight_regularization_slider_callback())
        self.weight_regularization_slider.set(0.0)
        self.Lambda = 0.0
        self.weight_regularization_slider.bind("<ButtonRelease-1>", lambda event: self.weight_regularization_slider_callback())
        self.weight_regularization_slider.grid(row=3, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        self.learning_method_variable = Tk.StringVar()
        self.learning_method_dropdown = Tk.OptionMenu(self.buttons_frame, self.learning_method_variable,"relu",
                                                      "sigmoid",
                                                      command=lambda event: self.learning_method_dropdown_callback())
        self.learning_method_variable.set("relu")
        self.activation = T.nnet.relu
        self.learning_method_dropdown.grid(row=1, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        self.adjust_weights_button = Tk.Button(self.buttons_frame,
                                               text="Adjust Weights (Train)",
                                               bg="yellow", fg="red",
                                               command=lambda: self.adjust_weights_button_callback())
        self.adjust_weights_button.grid(row=2, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        self.print_nn_parameters_button = Tk.Button(self.buttons_frame,
                                                    text="Print NN Parameters",
                                                    bg="yellow", fg="red",
                                                    command=lambda: self.print_nn_parameters_button_callback())
        self.print_nn_parameters_button.grid(row=3, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.x = []
        self.y = []

        self.current_epoch=0


    def hidden_layer_slider_callback(self):
        self.hidden_layer = self.hidden_layer_slider.get()

    def train_network_slider_callback(self):
        self.alpha = self.train_network_slider.get()

    def weight_regularization_slider_callback(self):
        self.Lambda  = self.weight_regularization_slider.get()
    def learning_method_dropdown_callback(self):
        self.learning_method = self.learning_method_variable.get()
        if self.learning_method == "relu": self.activation = T.nnet.relu
        else: self.activation = T.nnet.sigmoid

    # def refresh_display(self,mse_arr,mae_arr,start):
    #     x_axis = np.arange(start-1,start,1.0/len(mse_arr))
    #     self.x_axis_main.extend(x_axis)
    #     price_mse =[i[0] for i in mse_arr]
    #     volume_mse = [i[1] for i in mae_arr]
    #     price_mae = [i[0] for i in mae_arr]
    #     volume_mae = [i[1] for i in mae_arr]
    #     self.price_mse_main.extend(price_mse)
    #     self.price_mae_main.extend(price_mae)
    #     self.volume_mse_main.extend(volume_mse)
    #     self.volume_mae_main.extend(volume_mae)
    #     plt.plot(self.x_axis_main,self.price_mse_main,"r-",label="price_mse")
    #     plt.plot(self.x_axis_main,self.volume_mse_main,"b-",label="volume_mse")
    #     plt.plot(self.x_axis_main,self.price_mae_main,"y-",label="price_mae")
    #     plt.plot(self.x_axis_main,self.volume_mae_main,"g-",label="volume_mae")
    #     self.canvas.draw()

    def adjust_weights_button_callback(self):
        self.build_model()
        self.Train()

    def build_model(self):
        self.w1 = theano.shared(np.random.uniform(-1,1,size=(3072,self.hidden_layer)),name="w1")
        self.b1 = theano.shared(np.zeros((self.hidden_layer,)),name="b1")
        self.w2 = theano.shared(np.random.uniform(-1,1,size=(self.hidden_layer,10)),name="w2")
        self.b2 = theano.shared(np.zeros((10,)),name="b2")
        self.p = T.dmatrix("input")
        self.target = T.dmatrix("target")
        self.net1 = T.dot(self.p,self.w1) + self.b1
        #print theano.printing.debugprint(self.net1)
        self.a1 = self.activation(self.net1)
        self.net2 = T.dot(self.a1,self.w2) + self.b2
        self.a2 = T.nnet.softmax(self.net2)
        self.output = T.argmax(self.a2,axis=1)
        self.target_one = np.argmax(self.target,axis=1)
        self.cost = T.mean(T.nnet.categorical_crossentropy(self.a2,self.target)) + self.Lambda* (self.w1**2).sum() + self.Lambda * (self.w2**2).sum()
        self.error = T.mean(T.neq(self.output, self.target_one))

        self.dw1,self.db1,self.dw2,self.db2 = T.grad(self.cost,wrt=[self.w1,self.b1,self.w2,self.b2])
        self.train  = theano.function([self.p,self.target],self.error,
                                      updates=[[self.w1,self.w1 -self.alpha * self.dw1],
                                               [self.b1,self.b1 - self.alpha * self.db1],
                                               [self.w2,self.w2 - self.alpha * self.dw2],
                                               [self.b2,self.b2 - self.alpha * self.db2]])
        self.predict = theano.function(inputs=[self.p],outputs=self.output,on_unused_input="ignore")
        d3viz(self.cost,"sample.html")
    def Train(self):
        size = self.dataset.inputs.shape[0]
        self.err = []
        self.x = []
        for i in range(200):
            Errors =[]
            for j in range(0,size,32):
                #print self.dataset.targets[j:j+32].shape
                err= self.train(self.dataset.inputs[j:j+32],self.dataset.targets[j:j+32])
                Errors.append(err)
                self.err.append(np.mean(Errors))
                self.x.append(i)
            #self.display()
            print "epoch: {} Errors; {}".format(i,np.mean(Errors))
        self.display()
        self.Test()

    def Test(self):
        test_data = ClDataSet(TEST_FOLDER)
        y_pred = [0]* test_data.inputs.shape[0]
        for i in range(0,test_data.inputs.shape[0],32):
            y_pred[i:i+32] = self.predict(test_data.inputs[i:i+32])
        y_true = np.argmax(test_data.targets,axis=1)

        confusion_matrix = metrics.confusion_matrix(y_true,y_pred)

        norm_conf = []
        for i in confusion_matrix:
            a = 0
            tmp_arr = []
            a = sum(i, 0)
            for j in i:
                tmp_arr.append(float(j) / float(a))
            norm_conf.append(tmp_arr)

        fig = plt.figure()
        plt.clf()
        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                        interpolation='nearest')

        width, height = confusion_matrix.shape

        for x in xrange(width):
            for y in xrange(height):
                ax.annotate(str(confusion_matrix[x][y]), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center')

        cb = fig.colorbar(res)
        alphabet = '0123456789'
        plt.xticks(range(width), alphabet[:width])
        plt.yticks(range(height), alphabet[:height])
        plt.savefig('confusion_matrix.png', format='png')
    def display(self):
        plt.plot(self.x,self.err,"r")
        plt.savefig("task5.png")
        self.canvas.draw()


if __name__ == "__main__":
    nn_experiment_settings = {
        "min_initial_weights": -1.0,  # minimum initial weight
        "max_initial_weights": 1.0,  # maximum initial weight
        "number_of_inputs": 5,  # number of inputs to the network
        "train_network": 0.0001,  # learning rate
        "layers_specification": [{"number_of_neurons": 2, "activation_function": "linear"}],  # list of dictionaries
        'number_of_classes': 2,
         "weight_regularization":0.5,
        "delayed_element":5,
        "number_iterations":1,
        "hidden_layer":500,

        'number_of_samples_in_each_class': 3
    }

    main_frame = Tk.Tk()
    main_frame.title("back propagation")
    main_frame.geometry('650x760')
    ob_nn_gui_2d = ClNNGui2d(main_frame)
    main_frame.mainloop()
