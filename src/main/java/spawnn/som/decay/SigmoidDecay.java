package spawnn.som.decay;

import java.io.BufferedWriter;
import java.io.FileWriter;

public class SigmoidDecay extends DecayFunction {
	
	private double scale;
	
	public SigmoidDecay( double scale ) {
		this.scale = scale;
	}

	@Override
	public double getValue(double x) {
		
		return 1.0/(1.0+Math.exp( -x*scale+0.5*scale) );
	}
	
	public static void main(String[] args) {
		
		BufferedWriter bw = null;
		try {
			bw = new BufferedWriter( new FileWriter("output/test.csv") );
			
			SigmoidDecay gd = new SigmoidDecay( -20 );
			
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




