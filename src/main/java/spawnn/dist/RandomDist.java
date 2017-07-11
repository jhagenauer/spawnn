package spawnn.dist;

import java.util.Random;

public class RandomDist<T> implements Dist<T> {
	private Random r = new Random();
	
	@Override
	public double dist(T a, T b) {
		return r.nextDouble();
	}
}
