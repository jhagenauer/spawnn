package spawnn.som.decay;

import java.io.BufferedWriter;
import java.io.FileWriter;

public class ExponentialDecay extends DecayFunction {
	
	private double from, to, eta, span;
	
	public ExponentialDecay( double from, double to, double eta, double span) {
		this.from = from;
		this.to = to;
		this.eta = eta; // eta muss < 5 sein, gibt krümmung an
		this.span = span; // mit span < 1 geht Value ins neg.
	}

	@Override
	public double getValue(double x) {
		
		double g;
		if( eta <= 0 )
			eta = (g = Math.sqrt(to)) / (Math.sqrt(from) + g);
	    
		double offset = (eta * eta * (from - to) - to + 2. * eta * to) / (2.0 * eta - 1.0);
	    double f = Math.log( (from - offset) / (to - offset) ) / span;
	    g = Math.log(from - offset) / f;

		return  offset + Math.exp(f * (g - x ) );
	  
	}
	
	public static void main(String[] args) {
		
		BufferedWriter bw = null;
		try {
			bw = new BufferedWriter( new FileWriter("output/test.csv") );
			
			ExponentialDecay gd = new ExponentialDecay(0.85, 0.01, 0.1, 1.0 );
			
			bw.write("t,x\n");
			for( double d = 0 ; d <= 1.0 ; d+=0.01 ) 
				bw.write( d+","+gd.getValue(d)+"\n");
		
		} catch( Exception e ) {
			e.printStackTrace();
		} finally {
			try{ 
				bw.close(); 
			} catch( Exception e ) {
				e.printStackTrace();
			}
		}
	}
}




