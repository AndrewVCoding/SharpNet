using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace LinAlg
{
    public static class LinAlg
    {
        /// <summary>
        /// Computes the dot product between two matrices
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static Matrix Dot(Matrix a, Matrix b)
        {
            if (a.Size()[0] != b.Size()[0] || a.Size()[1] != b.Size()[1])
            {
                if (a.Size()[0] != b.Size()[1] | a.Size()[1] != b.Size()[0])
                    throw new Exception("Dot product: Mismatched dimensions: (" + a.Size()[0] + ", " + a.Size()[1] + ") + (" + b.Size()[0] + ", " + b.Size()[1] + ")");
                else
                    return Dot(a, b.T());
            }

            int rows = a.Size()[0];
            int cols = a.Size()[1];

            Matrix result = Matrix.Zero(1, 1);

            for(int i = 0; i < rows; i++)
                for(int j = 0; j < cols; j++)
                    result += a[i, j] * b[i, j];

            return result;
        }

        public static Matrix Sum(Matrix a, int axis = -1)
        {
            Matrix sum;

            if (axis == -1)
            {
                sum = new Matrix(1, 1);
                for (int i = 0; i < a.Size()[0]; i++)
                    for (int j = 0; j < a.Size()[1]; j++)
                        sum += a[i, j];
            }
            else if (axis == 0)
            {
                sum = new Matrix(1, a.Size()[1]);
                for (int i = 0; i < a.Size()[1]; i++)
                    sum[0, i] = Sum(a.GetCol(i))[0, 0];
            }
            else if (axis == 1)
            {
                sum = new Matrix(a.Size()[0], 1);
                for (int i = 0; i < a.Size()[0]; i++)
                    sum[i, 0] = Sum(a.GetRow(i))[0,0];
            }
            else
            {
                sum = new Matrix(a) * 0;
            }
            return sum;
        }

        public static Matrix Mean(Matrix a, int axis = -1)
        {
            Matrix mean;

            if (axis == -1)
            {
                mean = Sum(a) / a.Count();
                return mean;
            }
            else if (axis == 0)
            {
                mean = new Matrix(1, a.Size()[1]);
                for (int i = 0; i < a.Size()[1]; i++)
                    mean[0, i] = Sum(a.GetCol(i))[0,0] / a.Size()[0];
            }
            else if (axis == 1)
            {
                mean = new Matrix(a.Size()[0], 1);
                for (int i = 0; i < a.Size()[0]; i++)
                    mean[i, 0] = Sum(a.GetRow(i))[0, 0] / a.Size()[1];
            }
            else
                mean = new Matrix(a) * 0;
            return mean;
        }

        public static Matrix Exp(Matrix a)
        {
            Matrix result = new Matrix(a);
            for (int i = 0; i < a.Size()[0]; i++)
                for (int j = 0; j < a.Size()[1]; j++)
                    result[i, j] = Math.Exp(a[i, j]);

            return result;
        }

        public static Matrix Log(Matrix a)
        {
            Matrix result = new Matrix(a);
            for (int i = 0; i < a.Size()[0]; i++)
                for (int j = 0; j < a.Size()[1]; j++)
                    result[i, j] = Math.Log(a[i, j]);

            return result;
        }

        public static Matrix Hadamard(Matrix a, Matrix b)
        {
            if (!Enumerable.SequenceEqual(a.Size(), b.Size()))
            {
                if (a.Size()[0] == 1 && a.Size()[1] == 1)
                    return a * b;
                if (b.Size()[0] == 1 && b.Size()[1] == 1)
                    return b * a;
                else
                    throw new Exception("Hadamard product: Mismatched dimensions: (" + a.Size()[0] + ", " + a.Size()[1] + ") + (" + b.Size()[0] + ", " + b.Size()[1] + ")");
            }

            int rows = a.Size()[0];
            int cols = a.Size()[1];

            Matrix result = a;

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    result[i, j] = a[i, j] * b[i, j];

            return result;
        }

        public static Matrix Max(Matrix a, int axis = -1)
        {
            double max = a[0, 0];
            int[] index = { 0, 0 };
            Matrix result;

            if (axis == -1)
            {
                result = new Matrix(1, 2);
                for (int i = 0; i < a.Size()[0]; i++)
                    for (int j = 0; j < a.Size()[1]; j++)
                        if (a[i, j] > max)
                        {
                            max = a[i, j];
                            result[0,0] = i;
                            result[0,1] = j;
                        }
            }
            else if (axis == 0)
            {
                result = new Matrix(1, a.Size()[1]);
                for (int i = 0; i < a.Size()[1]; i++)
                    result[0, i] = Max(a.GetCol(i))[0, 0];
            }
            else if (axis == 1)
            {
                result = new Matrix(a.Size()[0], 1);
                for (int i = 0; i < a.Size()[0]; i++)
                    result[i, 0] = Max(a.GetRow(i))[0, 1];
            }
            else
            {
                throw new Exception("Invalid axis");
            }

            return result;
        }

        public static int[] SimpleMax(Matrix a)
        {
            double max = a[0, 0];
            int[] index = { 0, 0 };

            for (int i = 0; i < a.Size()[0]; i++)
                for (int j = 0; j < a.Size()[1]; j++)
                    if (a[i, j] > max)
                    {
                        max = a[i, j];
                        index[0] = i;
                        index[1] = j;
                    }
            return index;
        }
    }

    /// <summary>
    /// A matrix class.
    /// </summary>
    public class Matrix
    {
        // Number of decimals to print out when displaying the matrix.
        private static int precision = 2;

        // Threshold for Divide-and-Conquer matrix multiplaication algorithm
        private static int threshold = 500;

        private static Random random = new Random();

        private double[,] matrix;
        public int[] size;

        /// <summary>
        /// Constructs a matrix of zeros with the given dimensions.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        public Matrix(int a, int b)
        {
            matrix = new double[a, b];
            size = new int[] { a, b };
        }

        /// <summary>
        /// Constructs a matrix and populates its values from a 2-dimensional array.
        /// </summary>
        /// <param name="a"></param>
        public Matrix(double[,] a)
        {
            size = new int[] { a.GetLength(0), a.GetLength(1) };
            matrix = a;
        }

        /// <summary>
        /// Constructs a copy of the given matrix.
        /// </summary>
        /// <param name="a"></param>
        public Matrix(Matrix a)
        {
            matrix = new double[a.Size()[0], a.Size()[1]];

            for (int i = 0; i < a.Size()[0]; i++)
                for (int j = 0; j < a.Size()[1]; j++)
                    matrix[i, j] = a[i, j];

            size = a.size;
        }

        /// <summary>
        /// Returns an identity matrix of the given size.
        /// </summary>
        /// <param name="size"></param>
        /// <returns></returns>
        public static Matrix Ident(int size)
        {
            Matrix result = new Matrix(size, size);

            for (int i = 0; i < size; i++)
                result[i, i] = 1.0;
            return result;
        }

        /// <summary>
        /// Constructs a matrix of zeros with the given dimensions.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static Matrix Zero(int a, int b)
        {
            Matrix result = new Matrix(a, b);

            for (int i = 0; i < a; i++)
                for (int j = 0; j < b; j++)
                    result[i, j] = 0.0;
            return result;
        }

        /// <summary>
        /// Creates a random matrix of values between 0 and 1 with dimensions (a,b).
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static Matrix Rand(int a, int b)
        {
            Matrix result = new Matrix(a, b);

            for (int i = 0; i < a; i++)
                for (int j = 0; j < b; j++)
                    result[i, j] = random.NextDouble();

            return result;
        }

        /// <summary>
        /// Constructs a matrix of random values with a Gaussian probability distribution.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="mean"></param>
        /// <param name="sd"></param>
        /// <returns></returns>
        public static Matrix Gaussian(int a, int b, double mean = 0.0, double sd = 1.0)
        {
            Matrix result = new Matrix(a, b);

            for (int i = 0; i < a; i++)
                for (int j = 0; j < b; j++)
                {
                    double prob = Math.Exp(-1 * Math.Pow(0 - mean, 2) / (2 * sd * sd)) / Math.Pow(2 * Math.PI * sd * sd, 0.5);
                    double x = prob * random.NextDouble();

                    Boolean y = false;
                    while (!y)
                    {
                        double candidate = 2 * (random.NextDouble() - 0.5);
                        double candidate_prob = Math.Exp(-1 * Math.Pow(candidate - mean, 2) / (2 * sd * sd)) / Math.Pow(2 * Math.PI * sd * sd, 0.5);
                        if (candidate_prob >= x)
                        {
                            result[i, j] = candidate;
                            y = true;
                        }
                    }
                }

            return result;
        }

        /// <summary>
        /// Returns the total number of elements in the matrix.
        /// </summary>
        /// <returns></returns>
        public int Count()
        {
            return size[0] + size[1];
        }

        /// <summary>
        /// Returns the row at the given index.
        /// </summary>
        /// <param name="r"></param>
        /// <returns></returns>
        public Matrix GetRow(int r)
        {
            if (r < 0 || r > size[0])
                throw new IndexOutOfRangeException();

            Matrix row = new Matrix(1, size[1]);

            for(int i = 0; i < size[1]; i++)
                row[0, i] = matrix[r, i];

            return row;
        }

        /// <summary>
        /// Replaces the row at the given index.
        /// </summary>
        /// <param name="r"></param>
        /// <param name="target"></param>
        public void SetRow(int row, Matrix target)
        {
            if (size[1] == target.size[1])
                for(int j = 0; j < size[1]; j++)
                    this[row, j] = target[0, j];
        }

        /// <summary>
        /// Returns the column at the given index.
        /// </summary>
        /// <param name="c"></param>
        /// <returns></returns>
        public Matrix GetCol(int c)
        {
            if (c < 0 || c > size[1])
                throw new IndexOutOfRangeException();

            Matrix col = new Matrix(size[0], 1);

            for(int i = 0; i < size[0]; i++)
                col[i, 0] = matrix[i, c];

            return col;
        }

        /// <summary>
        /// Replaces the column at the specified index.
        /// </summary>
        /// <param name="col"></param>
        /// <param name="target"></param>
        public void SetCol(int col, Matrix target)
        {
            if (size[0] == target.size[0])
                for(int i = 0; i < size[0]; i++)
                    this[i, col] = target[i, 0];
        }

        /// <summary>
        /// Returns the length of the given dimension
        /// </summary>
        /// <param name="i"></param>
        /// <returns></returns>
        public int GetLength(int i)
        {
            return matrix.GetLength(i);
        }

        /// <summary>
        /// Returns the size of the matrix
        /// </summary>
        /// <returns></returns>
        public int[] Size()
        {
            return new int[] { size[0], size[1] };
        }

        /// <summary>
        /// Returns the transpose of the matrix
        /// </summary>
        /// <returns></returns>
        public Matrix T()
        {
            Matrix result = new Matrix(size[1], size[0]);

            Parallel.For(0, size[0], i =>
            {
                for (int j = 0; j < size[1]; j++)
                    result[j, i] = matrix[i, j];
            });
            return result;
        }

        /// <summary>
        /// Indexer for the matrix
        /// </summary>
        /// <param name="row"></param>
        /// <param name="col"></param>
        /// <returns></returns>
        public double this[int row, int col]
        {
            get
            {
                double val;
                if (row >= 0 && col >= 0 && row <= size[0] - 1 && col <= size[1] - 1)
                    val = matrix[row, col];
                else
                    val = 0;

                return val;
            }
            set
            {
                if (row >= 0 && col >= 0 && row <= size[0] - 1 && col <= size[1] - 1)
                    matrix[row, col] = value;
            }
        }

        /// <summary>
        /// Returns the matrix as a string for printing.
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            string result = "";

            for (int i = 0; i < size[0]; i++)
            {
                for (int j = 0; j < size[1]; j++)
                    result += String.Format("{0:N" + precision + "}", matrix[i, j]) + "\t";
                result += "\n";
            }

            return result;
        }

        /// <summary>
        /// Splits the matrix in half along the specified axis.
        /// Ex. Let A be a (4, 8) matrix.
        ///     Split(A, 0) -> A1 and A2, with dimensions (2, 8)
        ///     Split(A, 1) -> A1 and A2, with dimensions (4, 4)
        /// </summary>
        /// <param name="A"></param>
        /// <param name="axis"></param>
        /// <returns></returns>
        public static Matrix[] Split(Matrix A, int axis = 0)
        {
            Matrix[] result = new Matrix[2];

            // Split horizontally
            if (axis == 0)
            {
                int m = A.size[0] / 2;
                result[0] = new Matrix(m, A.size[1]);
                result[1] = new Matrix(A.size[0] - m, A.size[1]);

                Parallel.For(0, m, i =>
                {
                    result[0].SetRow(i, A.GetRow(i));
                    result[1].SetRow(i, A.GetRow(i + m));
                });
                result[1].SetRow(A.size[1] - m, A.GetRow(A.size[0] - 1));
            }

            // Split Vertically
            else if (axis == 1)
            {
                int m = A.size[1] / 2;
                result[0] = new Matrix(A.size[0], m);
                result[1] = new Matrix(A.size[0], A.size[1] - m);

                Parallel.For(0, m, i =>
                {
                    result[0].SetCol(i, A.GetCol(i));
                    result[1].SetCol(i, A.GetCol(i + m));
                });
                result[1].SetCol(A.size[1] - m, A.GetCol(A.size[1] - 1));
            }
            else
                throw new Exception("Invalid axis");

            return result;
        }

        /// <summary>
        /// Combines two matrices along the given axis.
        /// Ex. Let Matrices A and B have dimensions (3, 3)
        ///     Combine(A, B, 0) will have dimesnions (6, 3)
        ///     Combine(A, B, 1) will have dimensions (3, 6)
        /// </summary>
        /// <param name="A"></param>
        /// <param name="B"></param>
        /// <param name="axis"></param>
        /// <returns></returns>
        public static Matrix Combine(Matrix A, Matrix B, int axis)
        {
            Matrix result;
            // Combine horizontally
            if (axis == 0)
            {
                result = new Matrix(A.size[0] + B.size[0], A.size[1]);

                Parallel.For(0, A.size[0], i =>
                {
                    result.SetRow(i, A.GetRow(i));
                    result.SetRow(i + A.size[0], B.GetRow(i));
                });
                result.SetRow(A.size[0] + B.size[0] - 1, B.GetRow(B.size[0] - 1));
            }

            else if (axis == 1)
            {
                result = new Matrix(A.size[0], A.size[1] + B.size[1]);

                Parallel.For(0, A.size[1], i =>
                {
                    result.SetCol(i, A.GetCol(i));
                    result.SetCol(i + A.size[1], B.GetCol(i));
                });
                result.SetCol(A.size[1] + B.size[1] - 1, B.GetCol(B.size[1] - 1));
            }
            else
                throw new Exception("Invalid axis");

            return result;
        }

        // Operators

        /// <summary>
        /// Element-wise matrix addition.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static Matrix operator +(Matrix a, Matrix b)
        {
            if (!Enumerable.SequenceEqual(a.Size(), b.Size()))
                throw new Exception("Mismatched dimensions for matrix addition: (" + a.Size()[0] + ", " + a.Size()[1] + ") + (" + b.Size()[0] + ", " + b.Size()[1] + ")");

            Matrix result = new Matrix(a.size[0], a.size[1]);

            for(int i = 0; i < a.size[0]; i++)
                for (int j = 0; j < a.size[1]; j++)
                    result[i, j] = a[i, j] + b[i, j];

            return result;
        }

        /// <summary>
        /// Adds the value to every element in the matrix
        /// </summary>
        /// <param name="a"></param>
        /// <param name="B"></param>
        /// <returns></returns>
        public static Matrix operator +(double a, Matrix B)
        {
            Matrix result = new Matrix(B.size[0], B.size[1]);

            for(int i = 0; i < B.size[0]; i++)
                for (int j = 0; j < B.size[1]; j++)
                    result[i, j] = a + B[i, j];

            return result;
        }

        /// <summary>
        /// Adds the value to every element in the matrix
        /// </summary>
        /// <param name="a"></param>
        /// <param name="B"></param>
        /// <returns></returns>
        public static Matrix operator +(Matrix A, double b)
        {
            Matrix result = new Matrix(A.size[0], A.size[1]);

            for(int i = 0; i < A.size[0]; i++)
                for (int j = 0; j < A.size[1]; j++)
                    result[i, j] = b + A[i, j];

            return result;
        }

        /// <summary>
        /// Element-wise matrix subtraction.
        /// </summary>
        /// <param name="A"></param>
        /// <param name="B"></param>
        /// <returns></returns>
        public static Matrix operator -(Matrix A, Matrix B)
        {
            if (!Enumerable.SequenceEqual(A.Size(), B.Size()))
                throw new Exception("Mismatched dimensions for matrix subtraction: (" + A.Size()[0] + ", " + A.Size()[1] + ") - (" + B.Size()[0] + ", " + B.Size()[1] + ")");

            Matrix result = new Matrix(A.size[0], A.size[1]);

            for(int i = 0; i < A.size[0]; i++)
                for (int j = 0; j < A.size[1]; j++)
                    result[i, j] = A[i, j] - B[i, j];

            return result;
        }

        /// <summary>
        /// Element-wise subtraction of b - A
        /// </summary>
        /// <param name="b"></param>
        /// <param name="A"></param>
        /// <returns></returns>
        public static Matrix operator -(double b, Matrix A)
        {
            Matrix result = new Matrix(A.size[0], A.size[1]);

            for (int i = 0; i < A.size[0]; i++)
                for (int j = 0; j < A.size[1]; j++)
                    result[i, j] = b - A[i, j];

            return result;
        }

        /// <summary>
        /// Element-wise subtraction of a - b
        /// </summary>
        /// <param name="A"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static Matrix operator -(Matrix A, double b)
        {
            Matrix result = new Matrix(A.size[0], A.size[1]);

            for (int i = 0; i < A.size[0]; i++)
                for (int j = 0; j < A.size[1]; j++)
                    result[i, j] = A[i, j] - b;

            return result;
        }

        /// <summary>
        /// Matrix multiplication
        /// Can only multiply matrices with matching dimensions.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static Matrix operator *(Matrix A, Matrix B)
        {
            if (A.Size()[0] == 1 && A.Size()[1] == 1)
                return A[0, 0] * B;
            if (B.Size()[0] == 1 && B.Size()[1] == 1)
                return A * B[0, 0];

            if (A.Size()[1] != B.Size()[0])
                throw new Exception("Mismatched dimensions for matrix multiplication: (" + A.Size()[0] + ", " + A.Size()[1] + ") * (" + B.Size()[0] + ", " + B.Size()[1] + ")");

            Matrix result = new Matrix(A.Size()[0], B.Size()[1]);
            Matrix nmp = new Matrix(new double[,] { { A.size[0], A.size[1], B.size[1] } });
            int[] nmpMax = LinAlg.SimpleMax(nmp);

            // If below the threshold
            if (nmp[nmpMax[0], nmpMax[1]] < threshold)
            {
                for (int i = 0; i < A.Size()[0]; i++)
                    for (int j = 0; j < B.Size()[1]; j++)
                        for (int k = 0; k < A.Size()[1]; k++)
                            result[i, j] += A[i, k] * B[k, j];
            }

            else
            {
                // If n is the maximum, split A horizontally
                if (nmpMax[1] == 0)
                {
                    Matrix[] aSplit = Split(A, 0);
                    Parallel.For(0, 2, i =>
                    {
                        aSplit[i] = aSplit[i] * B;
                    });
                    result = Combine(aSplit[0], aSplit[1], 0);
                }

                // If p is the maximum, split B horizontally
                else if (nmpMax[1] == 2)
                {
                    Matrix[] bSplit = Split(B, 1);
                    Parallel.For(0, 2, i =>
                    {
                        bSplit[i] = A * bSplit[i];
                    });
                    result = Combine(bSplit[0], bSplit[1], 1);
                }

                else
                {
                    Matrix[] aSplit = Split(A, 1);
                    Matrix[] bSplit = Split(B, 0);
                    Parallel.For(0, 2, i =>
                    {
                        result += aSplit[i] * bSplit[i];
                    });
                    //result = aSplit[0] * bSplit[0] + aSplit[1] * bSplit[1];
                }
            }

            return result;
        }

        public static Matrix operator *(double a, Matrix b)
        {
            Matrix result = new Matrix(b.size[0], b.size[1]);

            for (int i = 0; i < b.size[0]; i++)
                for (int j = 0; j < b.size[1]; j++)
                    result[i, j] = a * b[i, j];

            return result;
        }

        public static Matrix operator *(Matrix a, double b)
        {
            Matrix result = new Matrix(a.size[0], a.size[1]);

            for (int i = 0; i < a.size[0]; i++)
                for (int j = 0; j < a.size[1]; j++)
                    result[i, j] = b * a[i, j];

            return result;
        }

        public static Matrix operator /(Matrix a, double b)
        {
            Matrix result = new Matrix(a.size[0], a.size[1]);

            for (int i = 0; i < a.size[0]; i++)
                for (int j = 0; j < a.size[1]; j++)
                    result[i, j] = a[i, j] / b;

            return result;
        }
    }
}