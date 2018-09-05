
namespace Roland.MatrixMath
{
    /// <summary>
    /// An interface for the mathematical matrix types.
    /// TODO: Kind of weak right now...
    /// </summary>
    public interface IMatrix
    {

        int RowLength { get; }
        int ColumnLength { get; }
        
    }
}
