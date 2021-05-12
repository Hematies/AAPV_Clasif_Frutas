# CODIGO REALIZADO A PARTIR DE AQUEL DE ESTE ENLACE:
# https://github.com/aaditkapoor/Fruit-Classifier-PyTorch/blob/master/fruit-classifier.py



import random
import torch
from torch import nn
from torch.autograd import Variable
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from commonQuant import *
from brevitas.export import FINNManager

gpu = 'cuda:0'
gpu = torch.device(gpu)

# In[99]:


# Preprocesamiento: Cambiar la resolución de las imágenes a 32x32:
transforms_train = T.Compose([T.RandomResizedCrop(32), T.ToTensor()])

# Cargamos los datos:
carpetaTraining = "../../Fruits_360-master/fruits/Training"
carpetaTest = "../../Fruits_360-master/fruits/Test"
image_data_train = ImageFolder(carpetaTraining,transform=transforms_train)
image_data_test = ImageFolder(carpetaTest,transform=transforms_train)

# Barajamos los datos:
random.shuffle(image_data_train.samples)
random.shuffle(image_data_test.samples)

# Número de clases:
classes_idx = image_data_train.class_to_idx
classes = len(image_data_train.classes)

# Número de imágenes de entrenamiento y test:
len_train_data = len(image_data_train)
len_test_data = len(image_data_test)


# Función que devuelve las etiquetas usadas en el dataset:
def get_labels():
    labels_train = []
    labels_test = []
    for i in image_data_train.imgs:
        labels_train.append(i[1])
    
    for j in image_data_test.imgs:
        labels_test.append(j[1])
    
    return (labels_train, labels_test)

labels_train, labels_test = get_labels()

# Definimos los cargadores de datos con un tamaño de batch de 100 imágenes:
batch_size = 100
train_loader = DataLoader(dataset=image_data_train,batch_size=batch_size)
test_loader = DataLoader(dataset=image_data_test,batch_size=batch_size)


print ("Tensor de entrada: ",iter(train_loader).next()[0].shape)

# Modelo CNN cuantizado:
class Model(nn.Module):

    def __init__(self, bit_width):
        super(Model, self).__init__()

        # Capas de obtención de características de la imagen:
        self.features = nn.Sequential(
            make_quant_conv2d(3, 64, kernel_size=3, stride=1,bit_width=bit_width, padding=0, groups=1,
                                       bias=False),
            make_quant_relu(bit_width),
            make_quant_max_pool(kernel_size=2, stride=2, padding=0),
            make_quant_conv2d(64, 64, kernel_size=4, stride=1,bit_width=bit_width, padding=0, groups=1,
                                       bias=False),
            make_quant_relu(bit_width),
            make_quant_max_pool(kernel_size=2,stride=2, padding=0),
            make_quant_conv2d(64, 64, kernel_size=3, stride=2,bit_width=bit_width, padding=0, groups=1,
                                       bias=False),
            make_quant_relu(bit_width),
            make_quant_max_pool(kernel_size=2,stride=2, padding=0))

        # Capas de clasificación de características finales:
        self.classifier = nn.Sequential(
            make_quant_linear(64, 100, bit_width=bit_width, bias=False),
            make_quant_relu(bit_width),
            make_quant_linear(100, classes, bit_width=bit_width, bias=False,
                              weight_scaling_per_output_channel=False))

    # Definimos la propagación del dato de entrada a través de la red:
    def forward(self, x):
        x = self.features(x)
        x = torch.reshape(x, (-1, 64))
        x = self.classifier(x)
        return x

# Número de bits con los que cuantizar:
numBits = 4

# Instanciamos el modelo:
model = Model(bit_width=numBits).to(gpu)

# Definimos el optimizador, la función de loss y el dispositivo donde se va a ejecutar el modelo:
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()
criterion.to(device=gpu)

# Función de entrenamiento del modelo:
def train(epochs):
    model.train()
    losses = []
    for epoch in range(1, epochs+1):
        print ("epoch #", epoch)
        current_loss = 0.0
        precisionEntrenamiento = 0.0
        for input, label in train_loader:
            input = input.to(gpu, non_blocking=True)
            label = label.to(gpu, non_blocking=True)

            x = Variable(input, requires_grad=False).float()
            y = Variable(label, requires_grad=False).long()

            optimizer.zero_grad() # Zeroing the grads
            y_pred = model(x)  # Calculating prediction

            correct = y_pred.max(1)[1].eq(y).sum()
            print ("no. of correct items classified (epoch = ",epoch,"): ", correct.item())
            precisionEntrenamiento = precisionEntrenamiento + correct.item()
            loss = criterion(y_pred, y) # Calculating loss (log_softmax already included)
            print ("loss: ", loss.item())
            current_loss+=loss.item()
            loss.backward() # Gradient cal
            optimizer.step() # Changing weights
        precisionEntrenamiento = precisionEntrenamiento / len_train_data
        losses.append(current_loss) # Only storing loss after every epoch

    print("Precisión de entrenamiento final: ", precisionEntrenamiento)
    return losses


# Función de testeo del modelo:
pred = []
def test():
    model.eval()
    with torch.no_grad():
        precisionTest = 0.0
        for input, label in test_loader:
            input = input.to(gpu, non_blocking=True)
            label = label.to(gpu, non_blocking=True)

            x = Variable(input, requires_grad=False).float()
            y = Variable(label, requires_grad=False).long()

            pred = model(x)
            #print ("acc: ", accuracy_score(labels_test, pred.max(1)[1].data.numpy()) * 100)
            print("acc: ", accuracy_score(y.data.cpu().numpy(), pred.max(1)[1].data.cpu().numpy()) * 100)
            precisionTest = precisionTest + accuracy_score(y.data.cpu().numpy(), pred.max(1)[1].data.cpu().numpy()) * 100
            loss = criterion(pred, y)
            print ("loss: ", loss.item())
        precisionTest = precisionTest /len_test_data
        print("Precisión de test final: ", precisionTest)



# Definimos el número de épocas y entrnamos:
numEpocas = 5 # OJO: Muchas épocas implica overfitting
train(numEpocas)

# Testeamos:
test()

# Exportamos el modelo (sus parámetros entrenables):
# model.load_state_dict(torch.load("modelos/modeloCuantizado_"+str(numBits)+"bits_sin_bias_arreglo_otra_red_nuevo.dat"))
torch.save(model.state_dict(), "modelos/modeloCuantizado_"+str(numBits)+"bits.dat")

# Exportamos el modelo con un fichero onnx (versión 1.5 de Onnx):
FINNManager.export(model.cpu(),
                    (1,3,32,32),
                export_path="modelos/modeloCuantizado_"+str(numBits)+"bits.onnx",
                   verbose=True)
