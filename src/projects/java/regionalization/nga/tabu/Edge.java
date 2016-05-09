package regionalization.nga.tabu;

import java.util.AbstractMap.SimpleEntry;
import java.util.Map.Entry;

public class Edge<T> {
	T a,b;
	
	Edge( Entry<T,T> e ) {
		this.a = e.getKey();
		this.b = e.getValue();
	}
	
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + a.hashCode();
		result = prime * result + b.hashCode();
		return result;
	}

	public T getA() {
		return a;
	}

	public T getB() {
		return b;
	}
	
	public Entry<T,T> asEntry() {
		return new SimpleEntry<T,T>(a,b);
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null || getClass() != obj.getClass())
			return false;
		Edge<T> other = (Edge<T>) obj;
		if( ( a == other.getA() && b == other.getB() ) || ( a == other.getB() && b == other.getA() ) )
			return true;
		return true;
	}
}
