package spawnn.dist;

import java.util.Map;

public class DistMapDist<T> implements Dist<T> {
	
	private Map<T,Map<T,Double>> m;
	
	public DistMapDist( Map<T,Map<T,Double>> m ) {
		this.m = m;
	}
	
	@Override
	public double dist(T a, T b) {
		return m.get(a).get(b);
	}
}
