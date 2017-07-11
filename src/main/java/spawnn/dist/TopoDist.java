package spawnn.dist;

import java.util.Collection;

import spawnn.ng.Connection;

public class TopoDist implements Dist<double[]> {
	
	Collection<Connection> cons;
	
	public TopoDist( Collection<Connection> cons ) {
		this.cons = cons;
	}
				
	@Override
	public double dist( double[] from, double[] to ) {	
		return Connection.dist(cons, from, to);
	}
}
