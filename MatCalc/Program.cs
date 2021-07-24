using System;
using System.Collections.Generic;
using System.Data;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using LinAlg;
using LumenWorks.Framework.IO.Csv;

class Program
{
    static Random random = new Random(1234);
    static System.Diagnostics.Stopwatch watch = new System.Diagnostics.Stopwatch();

    static void Main1(string[] args)
    {
        // Load the mnist data set
        Console.WriteLine("Loading Data...");
        watch.Restart();
        List<Matrix> data = LoadData("A:\\mnist\\mnist_train.csv", "A:\\mnist\\mnist_test.csv");
        Matrix train_s = data[0];
        Matrix train_y = data[1];
        Matrix test_s = data[2];
        Matrix test_y = data[3];
        Console.WriteLine("Data loaded: {0}ms", watch.ElapsedMilliseconds);

        // Create the neural network
        Console.WriteLine("initializing network...");
        watch.Restart();
        List<Layer> layers = new List<Layer>();
        layers.Add(new Dense(784, 100));
        layers.Add(new ReLU());
        layers.Add(new Dense(100, 200));
        layers.Add(new ReLU());
        layers.Add(new Dense(200, 10));
        Network network = new Network(layers, ErrorFunctions.CrossEntropy, ErrorFunctions.CrossEntropyGrad, 0.01);
        Console.WriteLine("network initialized in {0}ms\n", watch.ElapsedMilliseconds);

        // Train the network
        int batch_size = 32;
        int num_batches = 50;
        int num_epochs = 25;
        double epoch_test_accuracy = 0.0;

        for (int i = 1; i <= num_epochs; i++)
        {
            Console.WriteLine("Epoch {0}: ", i);
            train(network, batch_size, num_batches, train_s, train_y);
            epoch_test_accuracy = test(network, test_s, test_y);
            Console.WriteLine("");
        }
        Console.ReadLine();
    }

    public static void train(Network network, int batch_size, int num_batches, Matrix train_s, Matrix train_y)
    {
        double batch_loss = 0.0;
        long time = 0;

        // network.t = 1;

        Console.WriteLine("");

        for (int n = 1; n <= num_batches; n++)
        {
            // Seperate the batch inputs
            Matrix s = new Matrix(batch_size, train_s.Size()[1]);
            // Seperate the batch labels
            Matrix y = new Matrix(batch_size, train_y.Size()[1]);

            int row = Convert.ToInt32(random.NextDouble() * train_s.Size()[0]);
            for (int i = 0; i < batch_size; i++)
            {
                s.SetRow(i, train_s.GetRow(row));
                y.SetRow(i, train_y.GetRow(row));
                row = Convert.ToInt32(random.NextDouble() * train_y.Size()[0]);
            }

            watch.Restart();
            batch_loss += network.Train(s, y);
            watch.Stop();
            time += watch.ElapsedMilliseconds;

            Console.SetCursorPosition(0, Console.CursorTop - 1);
            drawTextProgressBar(n*batch_size, num_batches*batch_size);
            Console.WriteLine("Avg Loss: {0:N4}    time/batch: {1}ms", batch_loss/n, time/n);
        }
    }

    public static double test(Network network, Matrix test_s, Matrix test_y)
    {
        double num_correct = 0.0;
        int test_samples = 1000;
        int t = 0;
        Console.WriteLine("Testing\n");

        for(int i = 1; i <= test_samples / 32; i++)
        {
            Parallel.For(0, 32, n =>
            {
                int row = Convert.ToInt32(random.NextDouble() * (test_y.Size()[0] - 1));
                int prediction = Convert.ToInt16(network.OneHotPrediction(test_s.GetRow(row))[0, 0]);
                int answer = Convert.ToInt16(LinAlg.LinAlg.Max(test_y.GetRow(row), 1)[0, 0]);
                if (prediction == answer)
                    num_correct += 1.0;
                t++;
            });

            Console.SetCursorPosition(0, Console.CursorTop - 1);
            drawTextProgressBar(t, test_samples);
            Console.WriteLine("Accuracy: {0:N4}", num_correct / t);
        }

        return num_correct / (test_s.Size()[0] / test_samples);
    }

    public static List<Matrix> LoadData(string training_data, string test_data)
    {
        // Create the identity matrix for label encoding
        Matrix labs = Matrix.Ident(10);
        var traindataCSV = new DataTable();

        using (var csvReader = new CsvReader(new StreamReader(System.IO.File.OpenRead(@training_data)), true))
        {
            traindataCSV.Load(csvReader);
        }

        int total_stimuli = traindataCSV.Rows.Count;

        // Convert the training data into a matrix of inputs
        Matrix train_s = new Matrix(traindataCSV.Rows.Count, traindataCSV.Columns.Count - 1);
        Parallel.For(0, total_stimuli, i =>
        {
            for (int j = 1; j < traindataCSV.Columns.Count; j++)
                train_s[i, j] = Convert.ToDouble(traindataCSV.Rows[i][j].ToString()) / 255;
        });
        // Convert the training data into a matrix of labels
        Matrix train_y = new Matrix(traindataCSV.Rows.Count, 10);
        Parallel.For(0, traindataCSV.Rows.Count, i =>
        {
            object lab = traindataCSV.Rows[i][0];
            int row = Convert.ToInt16(traindataCSV.Rows[i][0].ToString());
            Matrix target = labs.GetRow(row);
            train_y.SetRow(i, target);
        });

        // Read the mnist test data
        Console.WriteLine("Loading test dataset...");
        var testdataCSV = new DataTable();
        using (var csvReader = new CsvReader(new StreamReader(System.IO.File.OpenRead(@test_data)), true))
        {
            testdataCSV.Load(csvReader);
        }
        // Convert the test data into a matrix of inputs
        Matrix test_s = new Matrix(testdataCSV.Rows.Count, testdataCSV.Columns.Count - 1);
        Parallel.For(0, testdataCSV.Rows.Count, i =>
        {
            for (int j = 1; j < testdataCSV.Columns.Count; j++)
                test_s[i, j] = Convert.ToDouble(testdataCSV.Rows[i][j].ToString()) / 255;
        });
        // Convert the training data into a matrix of labels
        Matrix test_y = new Matrix(testdataCSV.Rows.Count, 10);
        Parallel.For(0, testdataCSV.Rows.Count, i =>
        {
            object lab = testdataCSV.Rows[i][0];
            int row = Convert.ToInt16(testdataCSV.Rows[i][0].ToString());
            Matrix target = labs.GetRow(row);
            test_y.SetRow(i, target);
        });

        return new List<Matrix>() { train_s, train_y, test_s, test_y };
    }

    public static void ClearCurrentConsoleLine()
    {
        int currentLineCursor = Console.CursorTop;
        Console.SetCursorPosition(0, Console.CursorTop);
        Console.Write(new string(' ', Console.WindowWidth));
        Console.SetCursorPosition(0, currentLineCursor);
    }

    private static void drawTextProgressBar(int progress, int total)
    {
        //draw empty progress bar
        Console.CursorLeft = 0;
        Console.Write(""); //start
        Console.CursorLeft = 32;
        Console.Write(""); //end
        Console.CursorLeft = 1;
        float onechunk = 30.0f / total;

        //draw filled part
        int position = 1;
        for (int i = 0; i < onechunk * progress; i++)
        {
            Console.BackgroundColor = ConsoleColor.DarkGreen;
            Console.CursorLeft = position++;
            Console.Write(" ");
        }

        //draw unfilled part
        for (int i = position; i <= 31; i++)
        {
            Console.BackgroundColor = ConsoleColor.Red;
            Console.CursorLeft = position++;
            Console.Write(" ");
        }

        //draw totals
        Console.CursorLeft = 35;
        Console.BackgroundColor = ConsoleColor.Black;
        Console.Write(progress.ToString() + "/" + total.ToString() + "    "); //blanks at the end remove any excess
    }
}
