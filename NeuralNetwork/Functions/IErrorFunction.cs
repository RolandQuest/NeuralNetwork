using Roland.MatrixMath;

namespace Roland.NeuralNetwork
{
    /// <summary>
    /// Represents a function for calculating error.
    /// </summary>
    public interface IErrorFunction
    {
        double Error(double result, double expected);
        double DerivativeError(double result, double expected);
        Vector Error(Vector result, Vector expected);
        Vector DerivativeError(Vector result, Vector expected);
    }
}
