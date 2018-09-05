using System.Collections.Generic;
using System.Linq;
using Roland.MatrixMath;

namespace Roland.NeuralNetwork
{
    /// <summary>
    /// An ordered array of neurons.
    /// </summary>
    public class Layer : ILayer
    {
        /// <summary>
        /// An array object of the non-bias neurons in the layer.
        /// </summary>
        public Neuron[] OrderedNeurons { get; }

        /// <summary>
        /// The number of non-bias neurons in the layer.
        /// </summary>
        public int Length { get { return OrderedNeurons.Length; } }

        /// <summary>
        /// Constructs a new layer where all neurons use the given activation function and start with the default neuron value.
        /// </summary>
        /// <param name="af">The activation function to apply to all neurons.</param>
        /// <param name="length">The number of non-bias neurons in the layer.</param>
        public Layer(IActivationFunction af, int length)
        {
            OrderedNeurons = new Neuron[length];
            for (int i = 0; i < Length; i++)
                OrderedNeurons[i] = new Neuron(af);
        }

        /// <summary>
        /// Constructs a new layer based on previous neurons.
        /// Calls copy constructor on all passed neurons.
        /// </summary>
        /// <param name="neurons">An IEnumberable of neurons to be converted to list.</param>
        public Layer(IEnumerable<Neuron> neurons)
        {
            var neuronList = neurons.ToList();
            OrderedNeurons = new Neuron[neuronList.Count];
            for (int i = 0; i < Length; i++)
            {
                OrderedNeurons[i] = new Neuron(neuronList[i]);
            }
        }
        
        /// <summary>
        /// Gets a non-bias neuron in the layer.
        /// </summary>
        /// <param name="index">The index of non-bias neuron.</param>
        /// <returns>The neuron at given index.</returns>
        public Neuron this[int index]
        {
            get { return OrderedNeurons[index]; }
        }

        /// <summary>
        /// Sets the input values of all non-bias neurons.
        /// </summary>
        /// <param name="vals">A vector of values.</param>
        public void SetInputValues(Vector vals)
        {
            if(vals.Dimension != Length)
            {
                throw new System.Exception("Layer::SetInputValues(Vector) -> Dimensions do not match.");
            }

            for(int i = 0; i < Length; i++)
            {
                OrderedNeurons[i].InputValue = vals[i];
            }
        }

        /// <summary>
        /// Sets the input values of all non-bias neurons.
        /// </summary>
        /// <param name="vals">An IEnumberable of values.</param>
        public void SetInputValues(IEnumerable<double> vals)
        {
            var valsList = vals.ToList();
            if (valsList.Count != Length)
            {
                throw new System.Exception("Layer::SetInputValues(IEnumberable<double>) -> Dimensions do not match.");
            }

            for (int i = 0; i < Length; i++)
            {
                OrderedNeurons[i].InputValue = valsList[i];
            }
        }

        /// <summary>
        /// Sets the input values of all non-bias neurons.
        /// </summary>
        /// <param name="vals">A array of values.</param>
        public void SetInputValues(double[] vals)
        {
            if (vals.Length != Length)
            {
                throw new System.Exception("Layer::SetInputValues(double[]) -> Dimensions do not match.");
            }

            for (int i = 0; i < Length; i++)
            {
                OrderedNeurons[i].InputValue = vals[i];
            }
        }

        /// <summary>
        /// Get a vector of all the output values of non-bias neurons in layer.
        /// </summary>
        /// <returns>Vector representing layer outputs.</returns>
        public Vector GetOutputVector()
        {
            Vector v = new Vector(Length);
            for(int i = 0; i < Length; i++)
            {
                v[i] = OrderedNeurons[i].OutputValue;
            }
            return v;
        }

        /// <summary>
        /// Get a vector of all the input values of non-bias neurons in layer.
        /// </summary>
        /// <returns>Vector representing layer inputs.</returns>
        public Vector GetInputVector()
        {
            Vector v = new Vector(Length);
            for (int i = 0; i < Length; i++)
            {
                v[i] = OrderedNeurons[i].InputValue;
            }
            return v;
        }

        /// <summary>
        /// Get a vector of all the output error values of non-bias neurons in layer.
        /// </summary>
        /// <returns>Vector representing layer output errors.</returns>
        public Vector GetErrorOutVector()
        {
            Vector v = new Vector(Length);
            for (int i = 0; i < Length; i++)
            {
                v[i] = OrderedNeurons[i].ErrorOut;
            }
            return v;
        }

        /// <summary>
        /// Get a vector of all the output error values of non-bias neurons in layer.
        /// </summary>
        /// <returns>Vecotr representing layer output errors.</returns>
        public Vector GetErrorInVector()
        {
            Vector v = new Vector(Length);
            for (int i = 0; i < Length; i++)
            {
                v[i] = OrderedNeurons[i].ErrorIn;
            }
            return v;
        }
    }
}
