package spawnn.ng.sorter;

public abstract class SorterContext implements Sorter<double[]> {
	public abstract double[] getContext(double[] x ); 
}
