package spawnn.rbf;

import java.util.Map;
import java.util.Random;

import spawnn.dist.Dist;

public class AdaptIncRBF extends IncRBF {

	protected double oldError;
	protected double mod;

	public AdaptIncRBF(Map<double[], Double> hidden, double lrA, double lrB, Dist<double[]> distA, int aMax, double mod, double alpha, double beta, double delta, int out) {
		super( hidden, lrA, lrB, distA, aMax, alpha, beta, delta, out );
				
		if( new Random().nextBoolean() )
			this.mod = -mod;
		else
			this.mod = mod;
		this.oldError =  Double.MAX_VALUE;
	}

	public void adaptScale( double curError ) {
				
		if( curError > oldError ) // change direction of adaptation		
			mod = -mod;
		
		scale += mod;
		oldError = curError;
	}
}
