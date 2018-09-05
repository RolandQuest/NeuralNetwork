using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Roland.NeuralNetwork
{
    /// <summary>
    /// Represents the function f(x) = 1 / [ 1 + e^(-x) ]
    /// Sigmoid is a special case of the logistic function.
    /// </summary>
    public class SigmoidActivation : IActivationFunction
    {
        /// <summary>
        /// The function at the given domain value.
        /// </summary>
        /// <param name="input">The domain value.</param>
        /// <returns>The range value corresponding to the given domain value.</returns>
        public double At(double val)
        {
            double denominator = 1 + Math.Exp(-val);
            return 1 / denominator;
        }

        /// <summary>
        /// The derivative at the given domain value.
        /// </summary>
        /// <param name="x">The domain value.</param>
        /// <returns>The range value corresponding to the given domain value.</returns>
        public double DerivativeAt(double val)
        {
            double sigmoid = At(val);
            return sigmoid * (1 - sigmoid);
        }

    }
}
