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
            //training sessions.
            int Trials = 100;

            List<int> layerSizes = new List<int>() { 9, 4, 9};

            //Initializes a neural network with given layer sizes.
            //Each layer after the first uses the sigmoid activation function by default.
            //Network uses the squared error function by default.
            Network n = new Network(new Random(1), layerSizes);

            //Random input data of length equal to the first layerSize.
            Vector input = new Vector(new List<double>() { 1, 2, 3, 4, 5, 6, 7, 8, 9 });

            //Random binary ouput data of length equal to the last layerSize.
            Vector expected = new Vector(new List<double>() { 1, 0, 1, 0, 1, 0, 1, 0, 1 });

            double learningRate = 0.4;

            for(int trial = 0; trial < Trials; trial++)
            {
                n.SetInput(input);
                n.Fire();
                n.Learn(learningRate, expected);
            }

            PrintNetworkToConsole(n, expected);
        }

        public static void PrintNetworkToConsole(Network n, Vector expected)
        {
            Vector output = n.OutputLayerValues;

            for (int i = 0; i < output.Dimension; i++)
            {
                Console.Write(expected[i] + " -> ");
                Console.WriteLine("{0:F5}", output[i]);
            }
            Console.Write("-----------------\n");

            SquaredError se = new SquaredError();
            Console.Write("Error    ");
            Console.WriteLine("{0:F5}", se.TotalError(output, expected));
            Console.WriteLine();
            
            for (int layer = 0; layer < n.Depth - 1; layer++)
            {
                n.PrintLayerWeightsToConsole(layer);
                Console.WriteLine();
            }

            Console.Read();
        }
    }
}
