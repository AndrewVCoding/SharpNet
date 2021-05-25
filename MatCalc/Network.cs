using LinAlg;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LinAlg
{
    public class Network
    {
        public List<Layer> network = new List<Layer>();
        public double initial_learning_rate = 0.1;
        public int t = 0;

        public Network(List<Layer> layers, double initial_learning_rate = 0.1)
        {
            this.network = layers;
            this.initial_learning_rate = initial_learning_rate;
        }

        public List<Matrix> Forward(Matrix X)
        {
            List<Matrix> activations = new List<Matrix>();
            Matrix input = X;

            int i = 0;
            foreach (Layer layer in this.network)
            {
                activations.Add(layer.Forward(input));
                input = activations[activations.Count - 1];
                i++;
            }
            return activations;
        }

        public Matrix predict(Matrix input)
        {
            List<Matrix> activations = Forward(input);
            Matrix outputs = activations[activations.Count - 1];

            Matrix max = LinAlg.Max(outputs, 1);

            return max;
        }

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

            Matrix loss = ErrorFunctions.CrossEntropy(logits, y);
            Matrix loss_grad = ErrorFunctions.CrossEntropyGrad(logits, y);

            //loss = ErrorFunctions.MeansSquared(logits, y);
            //loss_grad = ErrorFunctions.MeansSquaredGrad(logits, y);

            for (int i = network.Count - 1; i >= 0; i--)
            {
                loss_grad = network[i].Backward(layer_inputs[i], loss_grad, learningRate());
            }

            double batch_loss = LinAlg.Mean(loss)[0,0];

            //Console.WriteLine(y.GetRow(0));
            //Console.WriteLine(logits.GetRow(0));

            t += input.Size()[0];

            return batch_loss;
        }

        public double learningRate()
        {
            return initial_learning_rate * (1.0 + t / 100.0) / (1.0 + Math.Pow(t, 2) / 60000.0);
            //return 0.1;
        }
    }

    public class Layer
    {
        public Layer() { }

        public virtual Matrix Forward(Matrix input)
        {
            return input;
        }

        public virtual Matrix Backward(Matrix input, Matrix grad_output, double gamma = 0.1)
        {
            Matrix d_layer_d_output = Matrix.Ident(input.Size()[0]);

            return grad_output;
        }
    }
    
    public class ReLU: Layer
    {
        public ReLU() { }

        public override Matrix Forward(Matrix input)
        {
            Matrix activation = ActivationFunctions.Relu(input);
            return activation;
        }

        public override Matrix Backward(Matrix input, Matrix grad_output, double gamma = 0.1)
        {
            Matrix result = new Matrix(input);
            for (int i = 0; i < input.Size()[0]; i++)
                for (int j = 0; j < input.Size()[1]; j++)
                    result[i, j] = Math.Max(0, input[i, j]) / input[i, j];

            //Matrix grad = LinAlg.Hadamard(grad_output, ActivationFunctions.D_Relu(input));
            Matrix grad = LinAlg.Hadamard(grad_output, result);
            return grad;
        }
    }

    public class Sigmoid : Layer
    {
        public Sigmoid() { }

        public override Matrix Forward(Matrix input)
        {
            Matrix activation = ActivationFunctions.Sigmoid(input);
            return activation;
        }

        public override Matrix Backward(Matrix input, Matrix grad_output, double gamma = 0.1)
        {
            Matrix grad = LinAlg.Hadamard(grad_output, ActivationFunctions.D_Sigmoid(input));
            return grad;
        }
    }

    public class Square : Layer
    {
        public Square() { }

        public override Matrix Forward(Matrix input)
        {
            return LinAlg.Hadamard(input, input);
        }

        public override Matrix Backward(Matrix input, Matrix grad_output, double gamma = 0.1)
        {
            return 2 * LinAlg.Hadamard(grad_output, input);
        }
    }

    public class Softmax: Layer
    {
        public Softmax() { }

        public override Matrix Forward(Matrix input)
        {
            Matrix exp = Matrix.Zero(input);

            for (int i = 0; i < exp.Size()[0]; i++)
                for (int j = 0; j < exp.Size()[1]; j++)
                    exp[i, j] = Math.Exp(input[i, j]);

            Matrix denom = LinAlg.Sum(exp, 1);

            Matrix activation = Matrix.Zero(input);

            for (int i = 0; i < exp.Size()[0]; i++)
                for (int j = 0; j < exp.Size()[1]; j++)
                    activation[i, j] = input[i, j] / denom[i,j];
            return activation;
        }

        public override Matrix Backward(Matrix input, Matrix grad_output, double gamma = 0.1)
        {
            return base.Backward(input, grad_output, gamma);
        }
    }

    public class Dense: Layer
    {
        public Matrix weights;
        private Matrix biases;

        public Dense(int input_units, int output_units)
        {
            this.weights = Matrix.Gaussian(input_units, output_units, 0, Math.Pow(2.0/(input_units + output_units), 0.5));
            this.biases = Matrix.Gaussian(1, output_units, 0, Math.Pow(2.0 / output_units, 0.5)) * 0;
        }

        public override Matrix Forward(Matrix input)
        {
            Matrix result = input * weights;
            for (int i = 0; i < result.Size()[0]; i++)
                for (int j = 0; j < result.Size()[1]; j++)
                    result[i, j] += biases[0, j];
            return result;
        }

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
