import math
import random
from tqdm import tqdm # Para crear una barra de progreso al momento de entrenar el modelo

def hypo(params: list, weights: list, bias: float) -> float:
    result = 0
    for i in range(len(params)):
        result += params[i] * weights[i]
    return result + bias

def error(y_pred:float,y_real:float)->float:
    return y_pred - y_real

def cost(y_hat:list, y:list)->float:
    result = 0
    for i in range(len(y)):
        result += (error(y_hat[i],y[i]))**2
    result = 1/len(y) * result
    return result

def r_squared(y_pred: list, y_real: list) -> float:
    n = len(y_real)
    mean_y = sum(y_real) / n
    ss_res = 0.0
    ss_tot = 0.0
    for i in range(n):
        ss_res += (y_real[i] - y_pred[i]) ** 2
        ss_tot += (y_real[i] - mean_y) ** 2
    return 1 - (ss_res / ss_tot)

def gradient(weights: list, bias: float, params: list, y: list):
    gradients = [0] * len(weights)
    bias_gradient = 0
    m = len(params)

    for i in range(m):
        predict = hypo(params[i], weights, bias)
        predict_error = error(predict, y[i])
        
        for j in range(len(weights)):
            gradients[j] += predict_error * params[i][j]
        bias_gradient += predict_error

    for j in range(len(weights)):
        gradients[j] = (2/m) * gradients[j]
    bias_gradient = (2/m) * bias_gradient
    
    return gradients, bias_gradient


def update(weights:list, bias:float, gradients:list, bias_gradient:float, lr:float):
    new_weights = []
    for i in range(len(weights)):
        new_weight = weights[i] - (lr * gradients[i])
        new_weights.append(new_weight)
    new_baias = bias - (lr * bias_gradient)
    return new_weights, new_baias


def predict(params:list, weights:list, bias:float):
    predictions = []
    for i in range(len(params)):
        predicton = hypo(params[i], weights, bias)
        predictions.append(predicton)
    return predictions


def train_step(weights:list, bias:float, params:list, y:list, lr:float):
    predictions = predict(params, weights, bias)
    current_cost = cost(predictions, y)
    gradients, bias_gradient = gradient(weights, bias, params, y)
    new_weights, new_bias = update(weights, bias, gradients, bias_gradient, lr)
    return new_weights, new_bias, current_cost

def train(params:list, y:list, validation_params:list, validation_y:list, lr:float, epochs:int, batch_size:int):
    # weights = [0 for _ in range(len(params[0]))]
    weights = [random.random() for _ in range(len(params[0]))]
    bias = 0
    train_costs = []
    validation_costs = []
    
    m = len(params)
    num_batches = math.ceil(m / batch_size)

    total_steps = epochs*num_batches
    pbar = tqdm(total=total_steps, desc="Entrenando modelo")
    
    for epoch in range(epochs):
        epoch_train_cost = 0    
        # Iteramos sobre los datos en pequeños lotes
        for i in range(num_batches):
            # Calculamos los índices de inicio y fin
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, m)
            # Extraemos el lote actual
            batch_params = params[start_idx:end_idx]
            batch_y = y[start_idx:end_idx]
            # Entrenamos sólo con este lote.
            weights, bias, batch_cost = train_step(weights, bias, batch_params, batch_y, lr)
            # Sumamos el costo del lote para sacar el promedio después
            epoch_train_cost += batch_cost
            # Actualizar barra de progreso
            pbar.update(1)
            pbar.set_postfix({
                "ep": f"{epoch+1}/{epochs}",
                "batch": f"{i+1}/{num_batches}",
            })
            
        # Costo promedio de entrenamiento de toda la época
        epoch_train_cost /= num_batches
        train_costs.append(epoch_train_cost)
        
        # Evaluamos en validación
        validation_predictions = predict(validation_params, weights, bias)
        validation_cost = cost(validation_predictions, validation_y)
        validation_costs.append(validation_cost)
        
    return weights, bias, train_costs, validation_costs


def run_regression_model(X_train,y_train,X_val,y_val,X_test,y_test,learning_rate=0.1,epochs=20,batch_size=2048):
    params = X_train.values.tolist()
    train_y = y_train.tolist()
    validation_params = X_val.values.tolist()
    validation_y = y_val.tolist()
    test_params = X_test.values.tolist()
    test_y = y_test.tolist()

    weights, bias, train_costs, validation_costs = train(params, train_y, validation_params, validation_y, learning_rate, epochs, batch_size)

    train_predictions = predict(params, weights, bias)
    validation_predictions = predict(validation_params, weights, bias)
    test_predictions = predict(test_params, weights, bias)
    predictions = {
        "train": train_predictions,
        "val": validation_predictions,
        "test": test_predictions,
        "y_train": train_y,
        "y_val": validation_y,
        "y_test": test_y
    }

    history = {
        "epoch": list(range(1,len(train_costs) + 1)),
        "train_mse": train_costs,
        "val_mse": validation_costs
    }

    model = {
        "type": "manual_linear_regression",
        "weights": weights,
        "bias": bias,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size
    }

    return model, predictions, history