using System;
using System.Collections.Generic;
using Roland.MatrixMath;

namespace Roland.NeuralNetwork
{
    /// <summary>
    /// The mean squared error (MSE) or mean squared deviation (MSD).
    /// </summary>
    public class SquaredError : IErrorFunction
    {
        /// <summary>
        /// Finds the individual element contribution to error.
        /// </summary>
        /// <param name="result">The element value.</param>
        /// <param name="expected">The expected value.</param>
        /// <returns>MSE of element.</returns>
        public double Error(double result, double expected)
        {
            return Math.Pow(expected - result, 2) / 2;
        }

        /// <summary>
        /// Returns the change in error with respect to the result value.
        /// </summary>
        /// <param name="result">The element value.</param>
        /// <param name="expected">The expected value.</param>
        /// <returns>Change in error with respect to the given result.</returns>
        public double DerivativeError(double result, double expected)
        {
            return result - expected;
        }

        /// <summary>
        /// Finds the error for all elements.
        /// </summary>
        /// <param name="result">A vector of element values.</param>
        /// <param name="expected">A vector of expected values.</param>
        /// <returns>Vector representing MSE for each element.</returns>
        public Vector Error(Vector result, Vector expected)
        {
            if(result.Dimension != expected.Dimension)
            {
                throw new Exception("SquaredError::Error(Vector,Vector) -> Dimensions do not match.");
            }

            Vector errors = new Vector(result.Dimension);

            for(int i = 0; i < result.Dimension; i++)
            {
                errors[i] = Error(result[i], expected[i]);
            }

            return errors;
        }

        /// <summary>
        /// The total error between the two vectors.
        /// </summary>
        /// <param name="result">A vector of element values.</param>
        /// <param name="expected">A vector of expected values.</param>
        /// <returns></returns>
        public double TotalError(Vector result, Vector expected)
        {
            Vector errors = Error(result, expected);

            double totalError = 0.0;
            for(int i = 0; i < errors.Dimension; i++)
            {
                totalError += errors[i];
            }

            return totalError;
        }

        /// <summary>
        /// Finds the change in error with respect to the result value for all elements.
        /// </summary>
        /// <param name="result">A vector of element values.</param>
        /// <param name="expected">A vector of expected values.</param>
        /// <returns>Vector representing change in error with respect to the given result.</returns>
        public Vector DerivativeError(Vector result, Vector expected)
        {
            if (result.Dimension != expected.Dimension)
            {
                throw new Exception("SquaredError::DerivativeError(Vector,Vector) -> Dimensions do not match.");
            }

            Vector errors = new Vector(result.Dimension);

            for (int i = 0; i < result.Dimension; i++)
            {
                errors[i] = DerivativeError(result[i], expected[i]);
            }

            return errors;
        }
    }
}
