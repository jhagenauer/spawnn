package spawnn.ng.sorter;


import java.util.List;

public interface Sorter <T> {
	public void sort( final T x, List<T> neurons );
}
