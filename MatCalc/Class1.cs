using System;
using System.Collections.Generic;
using System.Data;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using LinAlg;
using LumenWorks.Framework.IO.Csv;

class Test
{
    static void Main(string[] args)
    {
        MatrixMultTest();
    }

    public static void AutoEncoder()
    {
        Random random = new Random();
        int batch_size = 50;
        int num_batches = 1000;
        int num_samples = 50;

        int input_size = 20;
        int reduced_size = 60;

        // Generate random data
        Matrix A = Matrix.Rand(num_samples, input_size);
        Matrix B = Matrix.Rand(num_samples, input_size);

        List<Layer> dummy_layers = new List<Layer>();
        dummy_layers.Add(new Layer());

        List<Layer> layers = new List<Layer>();
        layers.Add(new Dense(input_size, 25));
        layers.Add(new Sigmoid());
        layers.Add(new Dense(25, reduced_size));
        layers.Add(new Sigmoid());
        layers.Add(new Dense(reduced_size, 25));
        layers.Add(new Sigmoid());
        layers.Add(new Dense(25, input_size));

        Network network = new Network(layers, ErrorFunctions.MeansSquared, ErrorFunctions.MeansSquaredGrad, 0.01);

        for (int i = 0; i < num_batches; i++)
        {
            Matrix data = Matrix.Zero(batch_size, input_size);
            for (int j = 0; j < batch_size; j++)
                data.SetRow(j, A.GetRow(random.Next(num_samples)));

            Console.WriteLine(network.Train(data, data));
        }

        Console.WriteLine(network.Loss(B, B));
        Console.ReadLine();
    }

    public static void MatrixMultTest()
    {
        var watch = new System.Diagnostics.Stopwatch();

        Console.WriteLine("initializing matrices...");
        double[,] a = { { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 }, { 0, 0, 0, 1 } };
        double[,] b = { { 1, 1, 1, 1 }, { 2, 2, 2, 2 }, { 3, 3, 3, 3 }, { 4, 4, 4, 4 } };


        Matrix A = new Matrix(a);
        Matrix B = new Matrix(b);

        Console.WriteLine(A * B);

        watch.Restart();

        A = Matrix.Zero(1000, 1000) + 1.0;
        B = Matrix.Zero(1000, 1000) + 1.0;

        Console.WriteLine("Matrices Initialized: {0}", watch.ElapsedMilliseconds);
        double avg = 0.0;
        double t = 0.0;
        Matrix C = new Matrix(A);
        //Console.ReadLine();
        for (int i = 0; i < 10; i++)
        {
            watch.Restart();
            C = A * B;
            t = watch.ElapsedMilliseconds;
            avg += t / 10.0;
            Console.WriteLine("A*B=C: {0:N2}ms", t);
        }
        Console.WriteLine("Average: {0:N2}ms", avg);
        // Console.WriteLine(C);

        Console.ReadLine();
    }

    public static void SpeedTest()
    {
        var watch = new System.Diagnostics.Stopwatch();

        Console.WriteLine("initializing network...");
        watch.Restart();
        // Create the neural network
        List<Layer> layers = new List<Layer>();
        layers.Add(new Dense(784, 99));
        layers.Add(new ReLU());
        layers.Add(new Dense(99, 99));
        layers.Add(new ReLU());
        layers.Add(new Dense(99, 10));
        //layers.Add(new Sigmoid());

        Network network = new Network(layers, ErrorFunctions.MeansSquared, ErrorFunctions.MeansSquaredGrad, 0.01);
        Console.WriteLine("network initialized in {0}ms", watch.ElapsedMilliseconds);

        Matrix data = Matrix.Rand(1, 784);
        Matrix data5 = Matrix.Rand(5, 784);
        Matrix data50 = Matrix.Rand(50, 784);
        Matrix data500 = Matrix.Rand(500, 784);

        Matrix y = Matrix.Rand(1, 10);
        Matrix y5 = Matrix.Rand(5, 10);
        Matrix y50 = Matrix.Rand(50, 10);
        Matrix y500 = Matrix.Rand(500, 10);

        watch.Restart();
        for (int i = 0; i < 500; i++)
            network.Train(data, y);
        Console.WriteLine("total: {0} ms     {1} ms/stimulus", watch.ElapsedMilliseconds, watch.ElapsedMilliseconds / 500);

        watch.Restart();
        for (int i = 0; i < 50; i++)
            network.Train(data5, y5);
        Console.WriteLine("total: {0} ms     {1} ms/stimulus", watch.ElapsedMilliseconds, watch.ElapsedMilliseconds / 500);

        watch.Restart();
        for (int i = 0; i < 5; i++)
            network.Train(data50, y50);
        Console.WriteLine("total: {0} ms     {1} ms/stimulus", watch.ElapsedMilliseconds, watch.ElapsedMilliseconds / 500);

        watch.Restart();
        for (int i = 0; i < 1; i++)
            network.Train(data500, y500);
        Console.WriteLine("total: {0} ms     {1} ms/stimulus", watch.ElapsedMilliseconds, watch.ElapsedMilliseconds / 500);

        Console.ReadLine();
    }
}
