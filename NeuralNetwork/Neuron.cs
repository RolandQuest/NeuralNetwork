
namespace Roland.NeuralNetwork
{
    /// <summary>
    /// A simple neuron for a neural network.
    /// </summary>
    public class Neuron : INeuron
    {

        private double _inputValue;
        private double _errorOut;

        /// <summary>
        /// The function acting on the input value to produce the output value.
        /// </summary>
        public IActivationFunction ActivationFunction { get; set; }

        /// <summary>
        /// The final value of the neuron after the activation function is applied.
        /// </summary>
        public double OutputValue { get; private set; }

        /// <summary>
        /// The initial value of the neuron.
        /// </summary>
        public double InputValue
        {
            get
            {
                return _inputValue;
            }
            set
            {
                _inputValue = value;

                //Guarantee input and output value are always in sync.
                OutputValue = ActivationFunction.At(_inputValue);
            }
        }

        /// <summary>
        /// The change in the network's total error with respect to this neuron's OutputValue.
        /// </summary>
        public double ErrorOut
        {
            get
            {
                return _errorOut;
            }
            set
            {
                _errorOut = value;

                //Guarantee ErrorOut and ErrorIn are in sync.
                ErrorIn = _errorOut * ActivationFunction.DerivativeAt(InputValue);
            }
        }

        /// <summary>
        /// The change in the network's total error with respect to this neuron's InputValue.
        /// </summary>
        public double ErrorIn { get; private set; }

        /// <summary>
        /// Basic constructor. Every neuron must have an activation function.
        /// </summary>
        /// <param name="af">Activation function applied to InputValue to produce OutputValue</param>
        /// <param name="inputVal">The initial InputValue. Default is 0.</param>
        public Neuron(IActivationFunction af, double inputVal = 0.0)
        {
            ActivationFunction = af;
            InputValue = inputVal;
        }
        
        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="copy">The neuron to be copied.</param>
        public Neuron(Neuron copy)
        {
            ActivationFunction = copy.ActivationFunction;
            InputValue = copy.InputValue;
            ErrorOut = copy.ErrorOut;
        }
    }
}
