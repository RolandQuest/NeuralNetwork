using System;
using System.Collections.Generic;
using Roland.MatrixMath;
using Roland.NeuralNetwork;

namespace DQN
{
    class Program
    {
        static void Main(string[] args)
        {
            int Trials = 1000;

            List<int> sizes = new List<int>() { 27, 10, 10, 10, 18, 9};
            Network n = new Network(new Random(1), sizes);

            Vector input = new Vector(new List<double>() { 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            Vector expected = new Vector(new List<double>() { 1, 0, 1, 0, 1, 0, 1, 0, 1 });

            double learningRate = 0.4;

            for(int trial = 0; trial < Trials; trial++)
            {
                n.SetInput(input);
                n.Fire();
                n.Learn(learningRate, expected);
            }
            
            Vector output = n.OutputLayerValues;

            for(int i = 0; i < output.Dimension; i++)
                Console.Write(output[i] + "\n");
            Console.Write("\n");
            Console.Read();

        }
    }
}
