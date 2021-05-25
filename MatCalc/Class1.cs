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
    static void Main1(string[] args)
    {
        MatrixMultTest();
    }

    public static void MatrixMultTest()
    {
        var watch = new System.Diagnostics.Stopwatch();

        Console.WriteLine("initializing matrices...");
        double[,] a = { { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 }, { 0, 0, 0, 1 } };
        double[,] b = { { 1, 1, 1, 1 }, { 2, 2, 2, 2 }, { 3, 3, 3, 3 }, { 4, 4, 4, 4 } };

        Matrix A = Matrix.Ident(100);
        Matrix B = Matrix.Zero(100, 100) + 1.0;
        //A = new Matrix(a);
        //B = new Matrix(b);
        Console.WriteLine("Matrices Initialized: {0}", watch.ElapsedMilliseconds);

        watch.Restart();
        Matrix C = A * B;
        Console.WriteLine("A*B=C: {0}", watch.ElapsedMilliseconds);
        Console.WriteLine(C);

        Console.ReadLine();
    }

    public static void network_test()
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

        Network network = new Network(layers, 0.1);
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
        for (int i = 0; i < 10; i++)
            network.Train(data50, y50);
        Console.WriteLine("total: {0} ms     {1} ms/stimulus", watch.ElapsedMilliseconds, watch.ElapsedMilliseconds / 500);

        watch.Restart();
        for (int i = 0; i < 1; i++)
            network.Train(data500, y500);
        Console.WriteLine("total: {0} ms     {1} ms/stimulus", watch.ElapsedMilliseconds, watch.ElapsedMilliseconds / 500);

        watch.Restart();
        for (int i = 0; i < 100; i++)
            network.Train(data5, y5);
        Console.WriteLine("total: {0} ms     {1} ms/stimulus", watch.ElapsedMilliseconds, watch.ElapsedMilliseconds / 500);

        watch.Restart();
        for (int i = 0; i < 500; i++)
            network.Train(data, y);
        Console.WriteLine("total: {0} ms     {1} ms/stimulus", watch.ElapsedMilliseconds, watch.ElapsedMilliseconds / 500);

        Console.ReadLine();
    }
}
