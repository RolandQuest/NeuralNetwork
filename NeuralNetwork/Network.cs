using System;
using System.Collections.Generic;
using Roland.MatrixMath;

namespace Roland.NeuralNetwork
{
    /// <summary>
    /// Represents a Feed-Forward neural network.
    /// </summary>
    public class Network : INetwork
    {
        private List<Layer> _layers = new List<Layer>();
        private List<LayerWeighting> _layerConnections = new List<LayerWeighting>();

        /// <summary>
        /// The number of layers in the network including input and output.
        /// </summary>
        public int Depth
        {
            get
            {
                return _layers.Count;
            }
        }

        /// <summary>
        /// Basic constructor. Does a lot of assuming on types.
        /// </summary>
        /// <param name="r">A Random object used to randomize weights.</param>
        /// <param name="layerSizes">The sizes of the desired layers.</param>
        public Network(Random r, List<int> layerSizes)
        {
            bool isFirst = true;

            foreach (var size in layerSizes)
            {
                if (isFirst)
                {
                    _layers.Add(new Layer(new LinearActivation(), size));
                    isFirst = false;
                }
                else
                {
                    _layers.Add(new Layer(new SigmoidActivation(), size));
                }
            }

            for (int i = 0; i < _layers.Count - 1; i++)
            {
                _layerConnections.Add(new LayerWeighting(_layers[i], _layers[i + 1]));
                _layerConnections[i].RandomizeAllWeights(r, -0.1, 0.1);
            }
        }

        /// <summary>
        /// Gets the initial input of the network.
        /// </summary>
        public Vector InputLayerValues
        {
            get
            {
                return _layers[0].GetOutputVector();
            }
        }

        /// <summary>
        /// Gets the final ouput vector of the network.
        /// </summary>
        public Vector OutputLayerValues
        {
            get
            {
                return _layers[_layers.Count - 1].GetOutputVector();
            }
        }

        /// <summary>
        /// Activates the network to get a result.
        /// </summary>
        public void Fire()
        {
            for (int coms = 0; coms < _layerConnections.Count; coms++)
            {
                _layerConnections[coms].ForwardPropogate();
            }
        }

        /// <summary>
        /// Lets the network learn from the experience.
        /// </summary>
        /// <param name="learningRate">The rate of learning for the network.</param>
        /// <param name="expected">The expected values of the network.</param>
        public void Learn(double learningRate, Vector expected)
        {
            //Calculate error for each Output Neuron.
            IErrorFunction ef = new SquaredError();
            var OmegaLayer = _layers[_layers.Count - 1];

            for(int i = 0; i < OmegaLayer.Length; i++)
            {
                OmegaLayer[i].ErrorOut = ef.DerivativeError(OmegaLayer[i].OutputValue, expected[i]);
            }
            
            for(int comIndex = _layerConnections.Count - 1; comIndex >= 0; comIndex--)
            {
                _layerConnections[comIndex].BackwardPropogate(learningRate);
            }
            
        }

        /// <summary>
        /// Sets the input of the network.
        /// </summary>
        /// <param name="vals">A vector of input values.</param>
        public void SetInput(Vector vals)
        {
            _layers[0].SetInputValues(vals);
        }
        
        /// <summary>
        /// Prints the weights of a layer to the next layer.
        /// The last layer has no such weights.
        /// </summary>
        /// <param name="layerIndex">The index of the layer.</param>
        public void PrintLayerWeightsToConsole(int layerIndex)
        {
            _layerConnections[layerIndex].PrintWeightsToConsole();
        }
    }
}
