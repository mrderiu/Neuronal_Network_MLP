# Neuronal Network

Deep learning is about building large scale neural networks. The simplest form of neural network is called multilayer perceptron model. The building block for neural networks are artificial neurons or perceptrons. These are simple computational units that have weighted input signals and produce an output signal using an activation function.

Perceptrons are arranged into networks. A row of perceptrons is called a layer and one network can have multiple layers. The architecture of the perceptrons in the network is often called the network topology. Once configured, the neural network needs to be trained on your dataset. The classical and still preferred training algorithm for neural networks is called stochastic gradient descent.

Use a standard binary (two-class) classification dataset from the UCI Machine Learning Repository, like the Pima Indians dataset1. To keep things simple, the network model is just a few layers of fully-connected perceptrons.
In this particular model, the dataset has 12 inputs or predictors and the output is a single value of 0 or 1. Therefore, the network model should have 12 inputs (at the first layer) and 1 output (at the last layer).

nn.Linear: esta clase implementa una transformación lineal de los datos de entrada. Toma el tamaño de entrada y el tamaño de salida como parámetros y calcula la salida como una matriz de multiplicación de la entrada y una matriz de peso aprendible, seguida de un término de sesgo opcional.
nn.ReLU: esta clase implementa la función de activación de la unidad lineal rectificada, que se usa comúnmente en las redes neuronales para introducir la no linealidad.

        model = nn.Sequential(
        nn.Linear(8, 12),
        nn.ReLU(),
        nn.Linear(12, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid()
        )

This is a network with 3 fully-connected layers. Each layer is created in PyTorch using the nn.Linear(x, y) syntax which the first argument is the number of input to the layer and the second is the number of output. Between each layer, a rectified linear activation is used, but at the output, sigmoid activation is applied such that the output value is between 0 and 1. This is a typical network. A deep learning model is to have a lot of such layers in a model.

Cuando entrenamos una red neuronal, generalmente necesitamos dos ingredientes:

Una función de coste que nos sirva para calcular el error del modelo. En este sentido, el módulo torch.nn cuenta con diferentes clases para calcular errores, siendo las más conocidas nn.CrossEntropyLoss (para clasificación) y nn.MSELoss para regresión. De todos modos, puedes encontrar todas las opciones posibles aquí.
Un optimizador que nos permita optimizar los parámetros de los modelos a la hora de aplicar el back-propagation de la capa de salida a las capas ocultas. Para ello, Pytorch cuenta con el módulo torch.optim, el cual engloba diferentes funciones de optimización, siendo los más usados Adam y SGD. Puedes encontrar todos los algortimos aquí.

        loss_fn = nn.BCELoss() # binary cross-entropy loss
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        n_epochs = 500
        batch_size = 5
        
        for epoch in range(n_epochs):
            for i in range(0, len(X), batch_size):
                Xbatch = X[i:i+batch_size]
                y_pred = model(Xbatch)
                ybatch = y[i:i+batch_size]
                
                loss = loss_fn(y_pred, ybatch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            print(f'Finished epoch {epoch}, latest loss {loss}')    
