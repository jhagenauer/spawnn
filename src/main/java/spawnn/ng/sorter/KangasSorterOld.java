package spawnn.ng.sorter;

import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import spawnn.dist.Dist;

public class KangasSorterOld<T> extends DefaultSorter<T> {
	
	private final Dist<T> b;
	private int l;
		
	public KangasSorterOld( Dist<T> a, Dist<T> b, int l ) {
		super(a);
		this.b = b;
		this.l = l;
		
		if( l < 1 )
			throw new RuntimeException("l < 1");
	}
		
	@Override
	public void sort( final T x, List<T> neurons ) {
		super.sort(x, neurons); // sort neurons according to dist a
				
		// sort sublist
		Collections.sort(neurons.subList(0, l ), new Comparator<T>() {
			@Override
			public int compare(T o1, T o2) {
				if( b.dist(o1, x) < b.dist(o2, x) ) 
					return -1;				
				else if( b.dist(o1, x) > b.dist(o2, x) ) 
					return 1;
				else 
					return 0; 
			}
		});
	}
}
