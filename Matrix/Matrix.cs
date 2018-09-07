using System;

namespace Roland.MatrixMath
{
    /// <summary>
    /// The mathematical matrix.
    /// A rank 2 array of doubles.
    /// </summary>
    public class Matrix : IMatrix
    {

        protected double[,] _matrix;

        /// <summary>
        /// The dimension of the first rank.
        /// </summary>
        public int RowLength { get { return _matrix.GetLength(0); } }

        /// <summary>
        /// The dimension of the second rank.
        /// </summary>
        public int ColumnLength { get { return _matrix.GetLength(1); } }

        /// <summary>
        /// Basic constructor. Default values are all 0.
        /// </summary>
        /// <param name="dim1">The number of rows.</param>
        /// <param name="dim2">The number of columns.</param>
        public Matrix(int dim1, int dim2)
        {
            _matrix = new double[dim1, dim2];

            for (int row = 0; row < RowLength; row++)
            {
                for (int column = 0; column < ColumnLength; column++)
                {
                    _matrix[row, column] = 0.0;
                }
            }
        }

        /// <summary>
        /// Copy constructor. Copies all values from the given matrix.
        /// </summary>
        /// <param name="mat">The matrix to be copied.</param>
        public Matrix(Matrix mat)
        {
            _matrix = new double[mat.RowLength, mat.ColumnLength];

            for (int row = 0; row < RowLength; row++)
            {
                for (int column = 0; column < ColumnLength; column++)
                {
                    _matrix[row, column] = mat[row,column];
                }
            }
        }
        
        /// <summary>
        /// Converts a vector to a matrix of dimensions [Dimension,1].
        /// </summary>
        /// <param name="vector">Vector to convert.</param>
        public Matrix(Vector vector)
        {
            _matrix = new double[vector.Dimension, 1];
            for(int i = 0; i < vector.Dimension; i++)
            {
                _matrix[i, 0] = vector[i];
            }
        }

        /// <summary>
        /// Allows easy access to elements in the array.
        /// </summary>
        /// <param name="row">The row of the element in the matrix.</param>
        /// <param name="column">The column of the element in the matrix.</param>
        /// <returns>The value of the matrix at [row,column].</returns>
        public double this[int row, int column]
        {
            get
            {
                return _matrix[row,column];
            }
            set
            {
                _matrix[row, column] = value;
            }
        }

        /// <summary>
        /// Does a matrix multiplication and returns the product.
        /// </summary>
        /// <param name="A">The left side matrix.</param>
        /// <param name="B">The right side matrix.</param>
        /// <returns>The product of A * B.</returns>
        public static Matrix operator *(Matrix A, Matrix B)
        {
            if (A.ColumnLength != B.RowLength)
            {
                throw new Exception("Matrix::operator(*)(Matrix,Matrix) -> A.ColumnLength != B.RowLength");
            }

            Matrix m = new Matrix(A.RowLength, B.ColumnLength);

            for (int row = 0; row < A.RowLength; row++)
            {
                for (int column = 0; column < B.ColumnLength; column++)
                {
                    m[row, column] = 0.0;
                    for (int pos = 0; pos < A.ColumnLength; pos++)
                    {
                        m[row, column] += A[row, pos] * B[pos, column];
                    }
                }
            }

            return m;
        }

        /// <summary>
        /// Does a matrix multiplication and returns the product.
        /// </summary>
        /// <param name="A">The left side matrix.</param>
        /// <param name="B">A vector assumed to be a matrix of dimensions [Dimension,1].</param>
        /// <returns>The vector representing the product of A * B.</returns>
        public static Vector operator *(Matrix A, Vector B)
        {
            if (A.ColumnLength != B.Dimension)
            {
                throw new Exception("Matrix::operator(*)(Matrix,Vector) -> A.ColumnLength != B.Dimension");
            }

            Vector v = new Vector(A.RowLength);

            for (int row = 0; row < A.RowLength; row++)
            {
                v[row] = 0.0;
                for (int pos = 0; pos < A.ColumnLength; pos++)
                {
                    v[row] += A[row, pos] * B[pos];
                }
            }

            return v;
        }

        /// <summary>
        /// Returns a matrix that is the transpose of the given matrix.
        /// </summary>
        /// <param name="mat">The matrix to be transposed.</param>
        /// <returns>The transpose matrix of mat.</returns>
        public static Matrix Transpose(Matrix mat)
        {
            Matrix m = new Matrix(mat.ColumnLength, mat.RowLength);

            for(int row = 0; row < m.RowLength; row++)
            {
                for(int column = 0; column < m.ColumnLength; column++)
                {
                    m[row, column] = mat[column, row];
                }
            }

            return m;
        }

        /// <summary>
        /// Prints the matrix to console.
        /// </summary>
        public void PrintToConsole()
        {
            for(int row = 0; row < RowLength; row++)
            {
                Console.Write("{0:F3}", _matrix[row, 0]);
                for(int column = 1; column < ColumnLength; column++)
                {
                    Console.Write("\t");
                    Console.Write("{0:F3}", _matrix[row, column]);
                }
                Console.WriteLine();
            }
        }
    }
}
