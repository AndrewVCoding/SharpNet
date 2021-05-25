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
    }

    /// <summary>
    /// A matrix class.
    /// </summary>
    public class Matrix
    {
        private static Random random = new Random();

        private double[,] matrix;
        public int rows;
        public int cols;

        /// <summary>
        /// Constructs a matrix of zeros with the given dimensions
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        public Matrix(int a, int b)
        {
            matrix = new double[a, b];
            rows = a;
            cols = b;
        }

        /// <summary>
        /// Constructs a copy of the given matrix
        /// </summary>
        /// <param name="a"></param>
        public Matrix(Matrix a)
        {
            matrix = new double[a.Size()[0], a.Size()[1]];

            for (int i = 0; i < a.Size()[0]; i++)
                for (int j = 0; j < a.Size()[1]; j++)
                    matrix[i, j] = a[i, j];

            rows = a.Size()[0];
            cols = a.Size()[1];
        }

        /// <summary>
        /// Provides a matrix from a 2-dimensional array
        /// </summary>
        /// <param name="a"></param>
        public Matrix(double[,] a)
        {
            rows = a.GetLength(0);
            cols = a.GetLength(1);

            matrix = a;
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

        public static Matrix Zero(int a, int b)
        {
            Matrix result = new Matrix(a, b);

            for (int i = 0; i < a; i++)
                for (int j = 0; j < b; j++)
                    result[i, j] = 0.0;
            return result;
        }

        public static Matrix Zero(Matrix a)
        {
            return Zero(a.Size()[0], a.Size()[1]);
        }

        /// <summary>
        /// Creates a random matrix of values between 0 and 1 with dimensions (a,b)
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

        public static Matrix Gaussian(int a, int b, double mean=0.0, double sd=1.0)
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

        public int Count()
        {
            return rows * cols;
        }

        /// <summary>
        /// Returns the row at the given index
        /// </summary>
        /// <param name="r"></param>
        /// <returns></returns>
        public Matrix GetRow(int r)
        {
            if (r < 0 || r > rows)
                throw new IndexOutOfRangeException();

            Matrix row = new Matrix(1, cols);

            for (int i = 0; i < cols; i++)
                row[0, i] = matrix[r, i];

            return row;
        }

        public void SetRow(int row, Matrix target)
        {
            if (cols == target.cols)
                for (int j = 0; j < cols; j++)
                    this[row, j] = target[0, j];
        }

        /// <summary>
        /// Returns the column at the given index
        /// </summary>
        /// <param name="c"></param>
        /// <returns></returns>
        public Matrix GetCol(int c)
        {
            if (c < 0 || c > cols)
                throw new IndexOutOfRangeException();

            Matrix col = new Matrix(rows, 1);

            for (int i = 0; i < rows; i++)
                col[i, 0] = matrix[i, c];

            return col;
        }

        public void SetCol(int col, Matrix target)
        {
            if (rows == target.rows)
                for (int i = 0; i < rows; i++)
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
            return new int[] { rows, cols };
        }

        /// <summary>
        /// Returns the transpose of the matrix
        /// </summary>
        /// <returns></returns>
        public Matrix T()
        {
            Matrix result = new Matrix(cols, rows);

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    result[j, i] = matrix[i, j];
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
                if (row >= 0 && col >= 0 && row <= rows - 1 && col <= cols - 1)
                    val = matrix[row, col];
                else
                    val = 0;

                return val;
            }
            set
            {
                if (row >= 0 && col >= 0 && row <= rows - 1 && col <= cols - 1)
                    matrix[row, col] = value;
            }
        }

        public override string ToString()
        {
            string result = "";

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                    result += String.Format("{0:0.00}", matrix[i, j]) + "\t";
                result += "\n";
            }

            return result;
        }

        public static List<Matrix> Split(Matrix A, int axis=0)
        {
            Matrix A1;
            Matrix A2;
            // Split horizontally
            if (axis == 0)
            {
                int m = A.rows / 2;
                A1 = new Matrix(m, A.cols);
                A2 = new Matrix(A.rows - m, A.cols);

                for (int i = 0; i < m; i++)
                    A1.SetRow(i, A.GetRow(i));
                for (int i = 0; i < A2.rows; i++)
                    A2.SetRow(i, A.GetRow(i + m));
            }

            else if (axis == 1)
            {
                int m = A.cols / 2;
                A1 = new Matrix(A.rows, m);
                A2 = new Matrix(A.rows, A.cols - m);

                for (int i = 0; i < m; i++)
                {
                    Matrix x = A.GetCol(i);
                    A1.SetCol(i, A.GetCol(i));
                }
                for (int i = 0; i < A2.cols; i++)
                {
                    Matrix x = A.GetCol(i);
                    A2.SetCol(i, A.GetCol(i + m));
                }
            }
            else
                throw new Exception("Invalid axis");

            return new List<Matrix>() { A1, A2 };
        }

        public static Matrix Combine(Matrix A, Matrix B, int axis)
        {
            Matrix result;
            // Combine horizontally
            if (axis == 0)
            {
                result = new Matrix(A.rows + B.rows, A.cols);

                for (int i = 0; i < A.rows; i++)
                    result.SetRow(i, A.GetRow(i));
                for (int i = 0; i < B.rows; i++)
                    result.SetRow(i + A.rows, B.GetRow(i));
            }

            else if (axis == 1)
            {
                result = new Matrix(A.rows, A.cols + B.cols);

                for (int i = 0; i < A.cols; i++)
                    result.SetCol(i, A.GetCol(i));
                for (int i = 0; i < B.cols; i++)
                    result.SetCol(i + A.cols, B.GetCol(i));
            }
            else
                throw new Exception("Invalid axis");

            return result;
        }

        // Operators

        public static Matrix operator +(Matrix a, Matrix b)
        {
            if (!Enumerable.SequenceEqual(a.Size(), b.Size()))
                throw new Exception("Mismatched dimensions for matrix addition: (" + a.Size()[0] + ", " + a.Size()[1] + ") + (" + b.Size()[0] + ", " + b.Size()[1] + ")");

            Matrix result = new Matrix(a.rows, a.cols);

            for (int i = 0; i < a.rows; i++)
                for (int j = 0; j < a.cols; j++)
                    result[i, j] = a[i, j] + b[i, j];

            return result;
        }

        public static Matrix operator +(double a, Matrix B)
        {
            Matrix result = new Matrix(B.rows, B.cols);

            for (int i = 0; i < B.rows; i++)
                for (int j = 0; j < B.cols; j++)
                    result[i, j] = a + B[i, j];

            return result;
        }

        public static Matrix operator +(Matrix A, double b)
        {
            Matrix result = new Matrix(A.rows, A.cols);

            for (int i = 0; i < A.rows; i++)
                for (int j = 0; j < A.cols; j++)
                    result[i, j] = b + A[i, j];

            return result;
        }

        public static Matrix operator -(Matrix A, Matrix B)
        {
            if (!Enumerable.SequenceEqual(A.Size(), B.Size()))
                throw new Exception("Mismatched dimensions for matrix subtraction: (" + A.Size()[0] + ", " + A.Size()[1] + ") - (" + B.Size()[0] + ", " + B.Size()[1] + ")");

            Matrix result = new Matrix(A.rows, A.cols);

            for (int i = 0; i < A.rows; i++)
                for (int j = 0; j < A.cols; j++)
                    result[i, j] = A[i, j] - B[i, j];

            return result;
        }

        public static Matrix operator -(double a, Matrix b)
        {
            Matrix result = new Matrix(b.rows, b.cols);

            for (int i = 0; i < b.rows; i++)
                for (int j = 0; j < b.cols; j++)
                    result[i, j] = a - b[i, j];

            return result;
        }

        public static Matrix operator -(Matrix a, double b)
        {
            Matrix result = new Matrix(a.rows, a.cols);

            for (int i = 0; i < a.rows; i++)
                for (int j = 0; j < a.cols; j++)
                    result[i, j] = a[i, j] - b;

            return result;
        }

        public static Matrix operator *(Matrix A, Matrix B)
        {
            int threshold = 100;

            if (A.Size()[0] == 1 && A.Size()[1] == 1)
                return A[0, 0] * B;
            if (B.Size()[0] == 1 && B.Size()[1] == 1)
                return A * B[0, 0];

            //if (A.Size()[1] != B.Size()[0])
            //    throw new Exception("Mismatched dimensions for matrix multiplication: (" + A.Size()[0] + ", " + A.Size()[1] + ") * (" + B.Size()[0] + ", " + B.Size()[1] + ")");

            Matrix result = new Matrix(A.Size()[0], B.Size()[1]);

            if (A.Size()[0] < threshold | A.Size()[1] < threshold | B.Size()[1] < threshold)
                Parallel.For(0, A.Size()[0], i =>
                {
                    for (int j = 0; j < B.Size()[1]; j++)
                        result[i, j] = LinAlg.Dot(A.GetRow(i), B.GetCol(j))[0, 0];
                });

            // Split A vertically and B horizontally
            else
            {
                List<Matrix> aSplit = Split(A, 1);
                List<Matrix> bSplit = Split(B, 0);
                Matrix A1 = aSplit[0];
                Matrix A2 = aSplit[1];
                Matrix B1 = bSplit[0];
                Matrix B2 = bSplit[1];

                Matrix C1 = A1 * B1;
                Matrix C2 = A2 * B2;

                result = C1 + C2;
            }

            return result;
        }

        public static Matrix operator *(double a, Matrix b)
        {
            Matrix result = new Matrix(b.rows, b.cols);

            for (int i = 0; i < b.rows; i++)
                for (int j = 0; j < b.cols; j++)
                    result[i, j] = a * b[i, j];

            return result;
        }

        public static Matrix operator *(Matrix a, double b)
        {
            Matrix result = new Matrix(a.rows, a.cols);

            for (int i = 0; i < a.rows; i++)
                for (int j = 0; j < a.cols; j++)
                    result[i, j] = b * a[i, j];

            return result;
        }

        public static Matrix operator /(Matrix a, double b)
        {
            Matrix result = new Matrix(a.rows, a.cols);

            for (int i = 0; i < a.rows; i++)
                for (int j = 0; j < a.cols; j++)
                    result[i, j] = a[i, j] / b;

            return result;
        }
    }
}