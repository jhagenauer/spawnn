package spawnn.ng.sorter;


import java.util.List;

public interface Sorter <T> {
	public void sort( final T x, List<T> neurons );
	// often you are just interested in the first element of a sorter list and getBMU is usually faster
	public T getBMU( final T x, List<T> neurons );
}
