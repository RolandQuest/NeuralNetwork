using System;
using Roland.MatrixMath;

namespace Roland.NeuralNetwork
{
    /// <summary>
    /// Represents the weights between two Layer objects.
    /// </summary>
    public class LayerWeighting : ILayerWeighting
    {
        /// <summary>
        /// The weights from a layer Alpha to a layer Beta.
        /// [alphaLayerNeuronIndex][betaLayerNeuronIndex] = weight
        /// </summary>
        public Matrix Weights { get; }

        /// <summary>
        /// The weights for this connections bias unit.
        /// [betaLayerNeuronIndex] = weight from bias unit
        /// </summary>
        public Vector BiasWeights { get; }

        /// <summary>
        /// Reference to the alpha layer.
        /// </summary>
        public Layer AlphaLayer { get; private set; }

        /// <summary>
        /// Reference to the beta layer.
        /// </summary>
        public Layer BetaLayer { get; private set; }

        /// <summary>
        /// Basic constructor. Builds weights based on the given layers.
        /// </summary>
        /// <param name="alphaLayer">The alpha (from) layer.</param>
        /// <param name="betaLayer">The beta (to) layer.</param>
        public LayerWeighting(Layer alphaLayer, Layer betaLayer)
        {
            Weights = new Matrix(alphaLayer.Length, betaLayer.Length);
            BiasWeights = new Vector(betaLayer.Length);
            AlphaLayer = alphaLayer;
            BetaLayer = betaLayer;

            for (int i = 0; i < alphaLayer.Length; i++)
            {
                for (int j = 0; j < betaLayer.Length; j++)
                {
                    Weights[i, j] = 1.0;
                }
            }

            for (int j = 0; j < BetaLayer.Length; j++)
            {
                BiasWeights[j] = 1.0;
            }
        }

        /// <summary>
        /// Sets the input values of the BetaLayer neurons weighted from the output values of the AlphaLayer neurons.
        /// </summary>
        public void ForwardPropogate()
        {
            Vector alphaOutputVec = AlphaLayer.GetOutputVector();
            Vector v = Matrix.Transpose(Weights) * alphaOutputVec;

            for (int betaIndex = 0; betaIndex < BetaLayer.Length; betaIndex++)
            {
                v[betaIndex] += 1.0 * BiasWeights[betaIndex];
            }

            BetaLayer.SetInputValues(v);
        }

        /// <summary>
        /// Updates the weights and the error of the alpha layer.
        /// </summary>
        public void BackwardPropogate(double learningRate)
        {
            //Sequence-coupled
            BackwardPropogateAlphaLayer();
            BackwardPropogateWeights(learningRate);
        }

        /// <summary>
        /// Updates the weights between these two layers based on the error of the Beta Layer ErrorIn.
        /// </summary>
        private void BackwardPropogateWeights(double learningRate)
        {
            Vector betaErrorIn = BetaLayer.GetErrorInVector();

            for (int alphaIndex = 0; alphaIndex < Weights.RowLength; alphaIndex++)
            {
                for (int betaIndex = 0; betaIndex < Weights.ColumnLength; betaIndex++)
                {
                    double deltaErrorRespectToWeight = betaErrorIn[betaIndex] * AlphaLayer[alphaIndex].OutputValue;
                    Weights[alphaIndex, betaIndex] -= learningRate * deltaErrorRespectToWeight;
                }
            }
            
            for (int betaIndex = 0; betaIndex < Weights.ColumnLength; betaIndex++)
            {
                double deltaErrorRespectToWeight = betaErrorIn[betaIndex] * 1.0;
                BiasWeights[betaIndex] -= learningRate * deltaErrorRespectToWeight;
            }
        }

        /// <summary>
        /// Updates the Alpha Layer ErrorOut based on the Beta Layer ErrorIn.
        /// </summary>
        private void BackwardPropogateAlphaLayer()
        {
            Vector betaErrorIn = BetaLayer.GetErrorInVector();

            for (int alphaIndex = 0; alphaIndex < Weights.RowLength; alphaIndex++)
            {
                double sumOfError = 0.0;
                for (int betaIndex = 0; betaIndex < Weights.ColumnLength; betaIndex++)
                {
                    sumOfError += betaErrorIn[betaIndex] * Weights[alphaIndex, betaIndex];
                }
                AlphaLayer[alphaIndex].ErrorOut = sumOfError;
            }
        }

        /// <summary>
        /// Randomizes all of the weights.
        /// Each weight is assigned a double value between [min,max).
        /// Meant to be used on initialization.
        /// </summary>
        /// <param name="r">The Random object.</param>
        /// <param name="min">The inclusive minimum value in range.</param>
        /// <param name="max">The exclusive maximum value in range.</param>
        public void RandomizeAllWeights(Random r, double min, double max)
        {
            for (int i = 0; i < Weights.RowLength; i++)
            {
                for (int j = 0; j < Weights.ColumnLength; j++)
                {
                    Weights[i, j] = GetRandomInRange(r, min, max);
                }
            }
            for (int j = 0; j < BiasWeights.Dimension; j++)
            {
                BiasWeights[j] = GetRandomInRange(r, min, max);
            }
        }

        /// <summary>
        /// Prints the weights matrix to console.
        /// </summary>
        public void PrintWeightsToConsole()
        {
            //Column Headers
            Console.Write("    ");
            for (int j = 0; j < Weights.ColumnLength; j++)
            {
                Console.Write("\t" + "    C" + (j + 1));
            }
            Console.WriteLine();

            //Bias
            Console.Write("Bias");
            for(int i = 0; i < BiasWeights.Dimension; i++)
            {
                Console.Write("\t");
                if(BiasWeights[i] >= 0)
                {
                    Console.Write(" ");
                }
                Console.Write("{0:F3}", BiasWeights[i]);
            }
            Console.WriteLine();
            
            //Neurons
            for(int i = 0; i < Weights.RowLength; i++)
            {
                Console.Write("Row " + i);
                for(int j = 0; j < Weights.ColumnLength; j++)
                {
                    Console.Write("\t");
                    if(Weights[i,j] >= 0)
                    {
                        Console.Write(" ");
                    }
                    Console.Write("{0:F3}", Weights[i, j]);
                }
                Console.WriteLine();
            }
        }

        /// <summary>
        /// Gets a random double between [min,max).
        /// This really shouldn't be here...
        /// </summary>
        /// <param name="r">The Random object.</param>
        /// <param name="min">The inclusive minimum value in range.</param>
        /// <param name="max">The exclusive maximum value in range.</param>
        /// <returns>A random double evenly likely in the range.</returns>
        private double GetRandomInRange(Random r, double min, double max)
        {
            double num = r.NextDouble() * (max - min);
            num += min;
            return num;
        }


    }
}
