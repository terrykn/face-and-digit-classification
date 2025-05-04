from utils.data_loader import stream_load_data
from models.perceptron import Perceptron
from models.neural_net import NeuralNetwork
from models.torch_net import TorchNet
from evaluate import evaluate_model, evaluate_torch_model, run_model_on_random_samples

# 10% increments
fractions = [0.1 * i for i in range(1, 11)]

def NeuralNet(in_dim, out_dim):
    return NeuralNetwork(in_dim, 128, 64, out_dim)

print("Loading digit data...")
X_digit_train, y_digit_train = stream_load_data("data/digitdata/trainingimages", "data/digitdata/traininglabels", (28, 28))
X_digit_test, y_digit_test = stream_load_data("data/digitdata/testimages", "data/digitdata/testlabels", (28, 28))
X_digit_train /= 1.0
X_digit_test /= 1.0

evaluate_model(Perceptron, X_digit_train, y_digit_train, X_digit_test, y_digit_test, fractions, "digit")
#print(results)
evaluate_model(NeuralNet, X_digit_train, y_digit_train, X_digit_test, y_digit_test, fractions, "digit")
evaluate_torch_model(TorchNet, X_digit_train, y_digit_train, X_digit_test, y_digit_test, fractions, "digit")

print("\nLoading face data...")
X_face_train, y_face_train = stream_load_data("data/facedata/facedatatrain", "data/facedata/facedatatrainlabels", (70, 60))
X_face_test, y_face_test = stream_load_data("data/facedata/facedatatest", "data/facedata/facedatatestlabels", (70, 60))
X_face_train /= 1.0
X_face_test /= 1.0

evaluate_model(Perceptron, X_face_train, y_face_train, X_face_test, y_face_test, fractions, "face")
evaluate_model(NeuralNet, X_face_train, y_face_train, X_face_test, y_face_test, fractions, "face")
evaluate_torch_model(TorchNet, X_face_train, y_face_train, X_face_test, y_face_test, fractions, "face")


#run_model_on_random_samples(NeuralNet, X_digit_train, y_digit_train, X_digit_test, y_digit_test, (28, 28), title_prefix="Digit")