using LinAlg;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LinAlg
{
    /// <summary>
    /// A simple feed-forward and back-propogation neural network.
    /// </summary>
    public class Network
    {
        // List of layers in the network
        public List<Layer> network = new List<Layer>();
        // Initial Learning rate of the network
        public double initial_learning_rate = 0.1;
        // Current step used to calculate step size
        public int t = 0;

        public delegate Matrix MatrixFunction(Matrix A, Matrix B);

        public MatrixFunction loss_function;
        public MatrixFunction loss_function_grad;

        /// <summary>
        /// Creates a neural network model with from a list of layers and an initial learning rate.
        /// </summary>
        /// <param name="layers"></param>
        /// <param name="initial_learning_rate"></param>
        public Network(List<Layer> layers, MatrixFunction loss_func, MatrixFunction loss_func_grad, double initial_learning_rate = 0.1)
        {
            this.network = layers;
            this.initial_learning_rate = initial_learning_rate;
            this.loss_function = loss_func;
            this.loss_function_grad = loss_func_grad;
        }

        /// <summary>
        /// Feed-forward pass of the network, returns the networks output for a given stimulus.
        /// </summary>
        /// <param name="X"></param>
        /// <returns></returns>
        public List<Matrix> Forward(Matrix X)
        {
            List<Matrix> activations = new List<Matrix>();
            Matrix input = X;

            foreach (Layer layer in this.network)
            {
                activations.Add(layer.Forward(input));
                input = activations[activations.Count - 1];
            }

            return activations;
        }

        /// <summary>
        /// Returns the index of the highest value in the networks output.
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public Matrix OneHotPrediction(Matrix input)
        {
            List<Matrix> activations = Forward(input);
            Matrix outputs = activations[activations.Count - 1];

            return LinAlg.Max(outputs, 1);
        }

        /// <summary>
        /// Trains the network on the given input and target data.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        public double Train(Matrix input, Matrix y)
        {
            List<Matrix> layer_activations = Forward(input);
            List<Matrix> layer_inputs = new List<Matrix>();
            layer_inputs.Add(input);
            
            foreach (Matrix activation in layer_activations)
            {
                layer_inputs.Add(activation);
            }

            Matrix logits = layer_inputs[layer_inputs.Count - 1];

            Matrix loss = ErrorFunctions.MeansSquared(logits, y);
            Matrix loss_grad = ErrorFunctions.MeansSquaredGrad(logits, y);

            loss = loss_function(logits, y);
            loss_grad = loss_function_grad(logits, y);

            for (int i = network.Count - 1; i >= 0; i--)
            {
                loss_grad = network[i].Backward(layer_inputs[i], loss_grad, StepSize());
            }

            double batch_loss = LinAlg.Mean(loss)[0,0];
            t++;

            return batch_loss;
        }

        public double Loss(Matrix input, Matrix labels)
        {
            List<Matrix> layer_activations = Forward(input);
            List<Matrix> layer_inputs = new List<Matrix>();
            layer_inputs.Add(input);

            foreach (Matrix activation in layer_activations)
            {
                layer_inputs.Add(activation);
            }

            Matrix logits = layer_inputs[layer_inputs.Count - 1];

            Matrix loss = ErrorFunctions.MeansSquared(logits, labels);

            double batch_loss = LinAlg.Mean(loss)[0, 0];

            return batch_loss;
        }

        /// <summary>
        /// Computes the current stepsize of the network.
        /// </summary>
        /// <returns></returns>
        public double StepSize()
        {
            return initial_learning_rate * (1.0 + t / 100.0) / (1.0 + Math.Pow(t, 2) / 60000.0);
        }
    }

    /// <summary>
    /// A dummy neural network layer.
    /// </summary>
    public class Layer
    {
        public Layer() { }

        /// <summary>
        /// Returns the layer output
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public virtual Matrix Forward(Matrix input)
        {
            return input;
        }

        /// <summary>
        /// Applies back-propogation to the layer and returns its gradient.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="grad_output"></param>
        /// <param name="gamma"></param>
        /// <returns></returns>
        public virtual Matrix Backward(Matrix input, Matrix grad_output, double gamma = 0.1)
        {
            return grad_output;
        }
    }
    
    /// <summary>
    /// Rectified Linear Unit activation layer.
    /// </summary>
    public class ReLU: Layer
    {
        public ReLU() { }

        /// <summary>
        /// Returns the layer output.
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public override Matrix Forward(Matrix input)
        {
            return ActivationFunctions.Relu(input); ;
        }

        /// <summary>
        /// Returns the gradient of the layer.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="grad_output"></param>
        /// <param name="gamma"></param>
        /// <returns></returns>
        public override Matrix Backward(Matrix input, Matrix grad_output, double gamma = 0.1)
        {
            Matrix result = new Matrix(input);
            for (int i = 0; i < input.Size()[0]; i++)
                for (int j = 0; j < input.Size()[1]; j++)
                    result[i, j] = Math.Max(0, input[i, j]) / input[i, j];

            return LinAlg.Hadamard(grad_output, result); ;
        }
    }

    /// <summary>
    /// Logistic Sigmoid activation layer with range (0,1).
    /// </summary>
    public class Sigmoid : Layer
    {
        public Sigmoid() { }

        /// <summary>
        /// Returns layer output.
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public override Matrix Forward(Matrix input)
        {
            Matrix activation = ActivationFunctions.Sigmoid(input);
            return activation;
        }

        /// <summary>
        /// Returns the gradient of the layer.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="grad_output"></param>
        /// <param name="gamma"></param>
        /// <returns></returns>
        public override Matrix Backward(Matrix input, Matrix grad_output, double gamma = 0.1)
        {
            return LinAlg.Hadamard(grad_output, ActivationFunctions.D_Sigmoid(input));
        }
    }

    /// <summary>
    /// Square activation layer.
    /// </summary>
    public class Square : Layer
    {
        public Square() { }

        /// <summary>
        /// Returns layer output.
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public override Matrix Forward(Matrix input)
        {
            return LinAlg.Hadamard(input, input);
        }

        /// <summary>
        /// Returns the gradient of the layer.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="grad_output"></param>
        /// <param name="gamma"></param>
        /// <returns></returns>
        public override Matrix Backward(Matrix input, Matrix grad_output, double gamma = 0.1)
        {
            return 2 * LinAlg.Hadamard(grad_output, input);
        }
    }

    /// <summary>
    /// Softmax activation layer.
    /// </summary>
    public class Softmax: Layer
    {
        public Softmax() { }

        /// <summary>
        /// Returns layer output.
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public override Matrix Forward(Matrix input)
        {
            Matrix exp = Matrix.Zero(input.size[0], input.size[1]);

            for (int i = 0; i < exp.Size()[0]; i++)
                for (int j = 0; j < exp.Size()[1]; j++)
                    exp[i, j] = Math.Exp(input[i, j]);

            Matrix denom = LinAlg.Sum(exp, 1);

            Matrix activation = Matrix.Zero(input.size[0], input.size[1]);

            for (int i = 0; i < exp.Size()[0]; i++)
                for (int j = 0; j < exp.Size()[1]; j++)
                    activation[i, j] = input[i, j] / denom[i,j];

            return activation;
        }

        /// <summary>
        /// Returns the gradient of the layer.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="grad_output"></param>
        /// <param name="gamma"></param>
        /// <returns></returns>
        public override Matrix Backward(Matrix input, Matrix grad_output, double gamma = 0.1)
        {
            return base.Backward(input, grad_output, gamma);
        }
    }

    /// <summary>
    /// Fully connected layer of hidden units.
    /// </summary>
    public class Dense: Layer
    {
        // Layer weights
        public Matrix weights;
        // Layer Biases
        private Matrix biases;

        /// <summary>
        /// A dense network layer with weight and bias parameters that are updated through back-propogation. Parameters are initialized using a gaussian probability distribution with a standard deviationn equal to the square root of 2 divided by the number of parameters.
        /// </summary>
        /// <param name="input_units"></param>
        /// <param name="output_units"></param>
        public Dense(int input_units, int output_units)
        {
            this.weights = Matrix.Gaussian(input_units, output_units, 0, Math.Pow(2.0/(input_units + output_units), 0.5));
            this.biases = Matrix.Gaussian(1, output_units, 0, Math.Pow(2.0 / output_units, 0.5));
        }

        /// <summary>
        /// Returns the output of the layer.
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public override Matrix Forward(Matrix input)
        {
            Matrix result = input * weights;
            for (int i = 0; i < result.Size()[0]; i++)
                for (int j = 0; j < result.Size()[1]; j++)
                    result[i, j] += biases[0, j];
            return result;
        }

        /// <summary>
        /// Updates the layer parameters and returns its gradient.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="grad_output"></param>
        /// <param name="gamma"></param>
        /// <returns></returns>
        public override Matrix Backward(Matrix input, Matrix grad_output, double gamma=0.1)
        {
            Matrix grad_input = grad_output * weights.T();

            Matrix grad_weights = input.T() * grad_output;
            Matrix grad_biases = LinAlg.Mean(grad_output, 0);

            this.weights -= gamma * grad_weights;
            this.biases -= gamma * grad_biases;

            return grad_input;
        }
    }
}
