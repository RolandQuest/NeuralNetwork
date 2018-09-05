using Roland.MatrixMath;

namespace Roland.NeuralNetwork
{
    /// <summary>
    /// A neural network.
    /// </summary>
    public interface INetwork
    {

        Vector InputLayerValues { get; }
        Vector OutputLayerValues { get; }

        void SetInput(Vector inp);
        void Fire();
        void Learn(double learningRate, Vector expected);

    }
}
