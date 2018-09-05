using System;
using System.Collections.Generic;
using System.Linq;

namespace Roland.MatrixMath
{
    /// <summary>
    /// A 1-rank array of doubles with special mathematical properties.
    /// </summary>
    public class Vector : IVector
    {

        private double[] _vector;

        /// <summary>
        /// The number of elements in the vector.
        /// </summary>
        public int Dimension { get { return _vector.Length; } }

        /// <summary>
        /// Basic constructor. Initializes all values to 0.
        /// </summary>
        /// <param name="dim">The number of elements in the vector.</param>
        public Vector(int dim)
        {
            _vector = new double[dim];
            for (int i = 0; i < Dimension; i++)
            {
                _vector[i] = 0.0;
            }
        }

        /// <summary>
        /// Basic constructor. Initializes all values to the passed IEnumerable.
        /// </summary>
        /// <param name="vec">The IEnumerable to be converted to vector form.</param>
        public Vector(IEnumerable<double> vec)
        {
            var vecList = vec.ToList();
            _vector = new double[vecList.Count];

            for(int i = 0; i < Dimension; i++)
            {
                _vector[i] = vecList[i];
            }
        }

        /// <summary>
        /// Basic constructor. Initializes all values the passed array.
        /// </summary>
        /// <param name="vec">The array to be converted to vector form.</param>
        public Vector(double[] vec)
        {
            _vector = new double[vec.Length];

            for (int i = 0; i < Dimension; i++)
            {
                _vector[i] = vec[i];
            }
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="vec">The vector to be copied exactly.</param>
        public Vector(Vector vec)
        {
            _vector = new double[vec.Dimension];
            for(int i = 0; i < Dimension; i++)
            {
                _vector[i] = vec[i];
            }
        }

        /// <summary>
        /// Direct access to elements of the vector.
        /// </summary>
        /// <param name="index">The index of the desired element.</param>
        /// <returns>The element at position index.</returns>
        public double this[int index]
        {
            get
            {
                return _vector[index];
            }
            set
            {
                _vector[index] = value;
            }
        }

        /// <summary>
        /// Element wise addition of two vectors.
        /// </summary>
        /// <param name="a">Left side vector.</param>
        /// <param name="b">Right side vector.</param>
        /// <returns>A new vector representing a + b.</returns>
        public static Vector operator+ (Vector a, Vector b)
        {
            if(a.Dimension != b.Dimension)
            {
                throw new Exception("Vector::operator(+) -> Dimensions are not equal.");
            }

            Vector v = new Vector(a.Dimension);
            for(int i = 0; i < a.Dimension; i++)
            {
                v[i] = a[i] + b[i];
            }
            return v;
        }

        /// <summary>
        /// Element wise subtraction of two vectors.
        /// </summary>
        /// <param name="a">Left side vector.</param>
        /// <param name="b">Right side vector.</param>
        /// <returns>A new vector respresenting a - b.</returns>
        public static Vector operator- (Vector a, Vector b)
        {
            if (a.Dimension != b.Dimension)
            {
                throw new Exception("Vector::operator(-) -> Dimensions are not equal.");
            }

            Vector v = new Vector(a.Dimension);
            for (int i = 0; i < a.Dimension; i++)
            {
                v[i] = a[i] - b[i];
            }
            return v;
        }

        /// <summary>
        /// Element wise multiplication of two vectors.
        /// </summary>
        /// <param name="a">Left side vector.</param>
        /// <param name="b">Right side vector.</param>
        /// <returns>A new vector.</returns>
        public static Vector HadamardProduct(Vector a, Vector b)
        {
            if (a.Dimension != b.Dimension)
            {
                throw new Exception("Vector::HadamardProduct -> Dimensions not equal.");
            }

            Vector v = new Vector(a.Dimension);
            for (int i = 0; i < v.Dimension; i++)
            {
                v[i] = a[i] * b[i];
            }

            return v;
        }

        /// <summary>
        /// Mathematical dot product of two vectors.
        /// </summary>
        /// <param name="a">Left side vector.</param>
        /// <param name="b">Right side vector.</param>
        /// <returns>A double value.</returns>
        public static double DotProduct(Vector a, Vector b)
        {
            if(a.Dimension != b.Dimension)
            {
                throw new Exception("Vector::DotProduct -> Dimensions not equal.");
            }

            double val = 0.0;
            for(int i = 0; i < a.Dimension; i++)
            {
                val += a[i] * b[i];
            }

            return val;
        }

        /// <summary>
        /// Returns a [Dimension x 1] matrix representation of this vector.
        /// </summary>
        /// <returns>Returns a [Dimension x 1] matrix representation of this vector.</returns>
        public Matrix GetMatrixRepresentation()
        {
            return new Matrix(this);
        }
    }
}
