package spawnn.dist;

import java.util.Map;

// symmetric
public class MatrixDist<T> implements Dist<T> {
	
	Map<T,Map<T,Double>> m;
	
	public MatrixDist(Map<T,Map<T,Double>> m) {
		this.m = m;
	}

	@Override
	public double dist(T a, T b) {
		return m.get(a).get(b);
	}

}
