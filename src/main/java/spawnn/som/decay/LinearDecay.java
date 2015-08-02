package spawnn.som.decay;

import java.io.BufferedWriter;
import java.io.FileWriter;

public class LinearDecay extends DecayFunction {

	private double from, to;

	public LinearDecay(double from, double to) {
		this.from = from;
		this.to = to;
	}

	@Override
	public double getValue(double x) { 
		return (to - from) * x + from;
	}
public static void main(String[] args) {
		
		BufferedWriter bw = null;
		try {
			bw = new BufferedWriter( new FileWriter("output/test.csv") );
			
			LinearDecay gd = new LinearDecay(1.0, 0.0 );
			
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
