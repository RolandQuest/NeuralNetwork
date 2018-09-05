
namespace Roland.NeuralNetwork
{
    /// <summary>
    /// Represents the function f(x) = x
    /// </summary>
    public class LinearActivation : IActivationFunction
    {
        /// <summary>
        /// The function at the given domain value.
        /// </summary>
        /// <param name="input">The domain value.</param>
        /// <returns>The range value corresponding to the given domain value.</returns>
        public double At(double x)
        {
            return x;
        }

        /// <summary>
        /// The derivative at the given domain value.
        /// </summary>
        /// <param name="x">The domain value.</param>
        /// <returns>The range value corresponding to the given domain value.</returns>
        public double DerivativeAt(double x)
        {
            return 1.0;
        }

    }
}
