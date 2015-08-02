package spawnn.ng.sorter;

import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import spawnn.dist.Dist;

public class DefaultSorter<T> implements Sorter<T> {
	
	protected final Dist<T> d;
	
	public DefaultSorter( Dist<T> d) {
		this.d = d;
	}
	
	@Override
	public void sort( final T x, List<T> neurons ) {
		Collections.sort(neurons, new Comparator<T>() {
			@Override
			public int compare(T o1, T o2) {
				if( d.dist(o1, x) < d.dist(o2, x) ) return -1;				
				else if( d.dist(o1, x) > d.dist(o2, x) ) return 1;
				else return 0; }
		});
	}
}
