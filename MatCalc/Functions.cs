using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LinAlg
{
    public static class ActivationFunctions
    {
        public static double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        public static Matrix Sigmoid(Matrix x)
        {
            Matrix result = new Matrix(x);

            for (int i = 0; i < x.Size()[0]; i++)
                for (int j = 0; j < x.Size()[1]; j++)
                    result[i,j] = 1 / (1 + Math.Exp(-x[i,j]));

            return result;
        }

        public static double D_Sigmoid(double x)
        {
            return 2 * Math.Exp(-x) / Math.Pow(1 + Math.Exp(-x), 2);
        }

        public static Matrix D_Sigmoid(Matrix x)
        {
            Matrix result = new Matrix(x);

            for (int i = 0; i < x.Size()[0]; i++)
                for (int j = 0; j < x.Size()[1]; j++)
                    result[i, j] = 2 * Math.Exp(-x[i,j]) / Math.Pow(1 + Math.Exp(-x[i,j]), 2);

            return result;
        }

        public static double Relu(double x)
        {
            if (x <= 0)
                return 0;
            else
                return x;
        }

        public static Matrix Relu(Matrix x)
        {
            Matrix result = new Matrix(x);

            for (int i = 0; i < x.Size()[0]; i++)
                for (int j = 0; j < x.Size()[1]; j++)
                    if (x[i, j] <= 0)
                        result[i, j] = 0;
                    else
                        result[i, j] = x[i, j];

            return result;
        }

        public static double D_Relu(double x)
        {
            if (x <= 0)
                return 0;
            else
                return 1;
        }

        public static Matrix D_Relu(Matrix x)
        {
            Matrix result = new Matrix(x);

            for (int i = 0; i < x.Size()[0]; i++)
                for (int j = 0; j < x.Size()[1]; j++)
                    if (x[i, j] <= 0)
                        result[i, j] = 0;
                    else
                        result[i, j] = 1;

            return result;
        }

        public static double Softplus(double x)
        {
            return Math.Log(1 + Math.Exp(x));
        }

        public static Matrix Softplus(Matrix x)
        {
            Matrix result = new Matrix(x);

            for (int i = 0; i < x.Size()[0]; i++)
                for (int j = 0; j < x.Size()[1]; j++)
                    result[i, j] = Math.Log(1 + Math.Exp(x[i,j]));

            return result;
        }

        public static double D_Softplus(double x)
        {
            return Math.Exp(x) / (1 + Math.Exp(x));
        }

        public static Matrix D_Softplus(Matrix x)
        {
            Matrix result = new Matrix(x);

            for (int i = 0; i < x.Size()[0]; i++)
                for (int j = 0; j < x.Size()[1]; j++)
                    result[i, j] = Math.Exp(x[i,j]) / (1 + Math.Exp(x[i,j]));

            return result;
        }
    }

    public static class ErrorFunctions
    {
        public static double MeansSquared(double x, double y)
        {
            return Math.Pow(x - y, 2);
        }

        public static double MeansSquared(Matrix a)
        {
            for (int i = 0; i < a.Size()[0]; i++)
                for (int j = 0; j < a.Size()[1]; j++)
                    a[i, j] = a[i, j] * a[i,j];

            double result = (1.0 / a.Size()[0]) * LinAlg.Sum(a)[0,0];

            return result;
        }

        public static Matrix MeansSquared(Matrix a, Matrix b)
        {
            Matrix difference = a - b;

            Matrix result = LinAlg.Hadamard(difference, difference);

            return result;
        }

        public static Matrix MeansSquaredGrad(Matrix a, Matrix b)
        {
            Matrix result = -2 * (a - b);

            return result;
        }

        public static Matrix CrossEntropy(Matrix logits, Matrix labels)
        {
            Matrix logits_for_answers = new Matrix(logits.Size()[0], 1);
            for (int i = 0; i < logits.Size()[0]; i++)
                logits_for_answers[i,0] = Convert.ToDouble(LinAlg.Dot(logits.GetRow(i), labels.GetRow(i))[0,0]);
            
            Matrix logsum = LinAlg.Log(LinAlg.Sum(LinAlg.Exp(logits), 1));

            Matrix xentropy = logsum - logits_for_answers;

            return logsum - logits_for_answers;
        }

        public static Matrix CrossEntropyGrad(Matrix a, Matrix b)
        {
            Matrix result = LinAlg.Exp(a);

            Matrix denom = LinAlg.Sum(result, 1);

            for (int i = 0; i < result.Size()[0]; i++)
                for (int j = 0; j < result.Size()[1]; j++)
                    result[i, j] = result[i, j] / denom[i, 0];

            result = (result - b) / a.Size()[0];

            return result;
        }
    }
}
