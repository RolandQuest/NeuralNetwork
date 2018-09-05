
namespace Roland.NeuralNetwork
{
    /// <summary>
    /// Interface for a single neuron in a neural network.
    /// </summary>
    public interface INeuron
    {

        double InputValue { get; set; }
        double OutputValue { get; }
        double ErrorOut { get; }
        double ErrorIn { get; }

        IActivationFunction ActivationFunction { set; }
    }
}
