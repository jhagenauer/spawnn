package spawnn.ng.sorter;

import java.util.List;

import spawnn.dist.Dist;

public class KangasSorter<T> implements Sorter<T> {
	
	private final Sorter<T> a, b;
	private int l;
		
	public KangasSorter( Dist<T> a, Dist<T> b, int l ) {
		this(new DefaultSorter<T>(a), new DefaultSorter<T>(b), l);
	}
	
	public KangasSorter( Sorter<T> a, Sorter<T> b, int l ) {
		this.a = a;
		this.b = b;
		this.l = l;
		
		if( l < 1 )
			throw new RuntimeException("l < 1");
	}
		
	@Override
	public void sort( final T x, List<T> neurons ) {
		a.sort(x, neurons);
		b.sort(x, neurons.subList(0, l ));		
	}
}

