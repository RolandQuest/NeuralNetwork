using Roland.MatrixMath;

namespace Roland.NeuralNetwork
{
    /// <summary>
    /// Represents a matrix of weights between two layers.
    /// Handles updates between layers.
    /// </summary>
    public interface ILayerWeighting
    {

        Matrix Weights { get; }
        void ForwardPropogate();
        void BackwardPropogate(double learningRate);
        
    }
}
