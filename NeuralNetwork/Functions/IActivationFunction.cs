
namespace Roland.NeuralNetwork
{
    /// <summary>
    /// Interface for a function.
    /// </summary>
    public interface IActivationFunction
    {

        double At(double x);
        double DerivativeAt(double x);

    }
}
