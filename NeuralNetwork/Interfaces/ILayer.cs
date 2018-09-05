using Roland.MatrixMath;
using System.Collections.Generic;

namespace Roland.NeuralNetwork
{
    /// <summary>
    /// Represents an ordered array of Neurons.
    /// </summary>
    public interface ILayer
    {

        Neuron[] OrderedNeurons { get; }
        int Length { get; }

        void SetInputValues(Vector vals);
        void SetInputValues(IEnumerable<double> vals);
        void SetInputValues(double[] vals);

        Vector GetOutputVector();
        Vector GetErrorOutVector();

        Vector GetInputVector();
        Vector GetErrorInVector();

    }
}
