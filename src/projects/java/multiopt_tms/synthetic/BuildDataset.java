package multiopt_tms.synthetic;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import spawnn.utils.DataUtils;

public class BuildDataset {
	public static void main(String[] args) {
		Random r = new Random();
		
		List<double[]> samples = new ArrayList<double[]>();
		int clusterSize = 200;
		
		for( int i = 0; i < clusterSize; i++ ) {
			double[] d = new double[]{
				r.nextDouble(), 	// x
				r.nextDouble(), 	// y
				0.0,				// v
				0.0					// c
			};
			samples.add(d);
		}
		
		for( int i = 0; i < clusterSize; i++ ) {
			double[] d = new double[]{
				1,				 	// x
				1,					// y
				r.nextDouble(),		// v
				1.0					// c
			};
			samples.add(d);
		}
		
		for( int i = 0; i < clusterSize; i++ ) {
			double[] d = new double[]{
				0,				 		// x
				0,				 		// y
				r.nextDouble(),			// v
				2.0						// c
			};
			samples.add(d);
		}
		
		for( int i = 0; i < clusterSize; i++ ) {
			double[] d = new double[]{
				r.nextDouble()*0.3,		// x
				1-r.nextDouble()*0.3, 	// y
				1 - r.nextDouble()*0.3,	// v
				3.0						// c
			};
			samples.add(d);
		}
		
		for( int i = 0; i < clusterSize; i++ ) {
			double[] d = new double[]{
				1-r.nextDouble()*0.3, 	// x
				r.nextDouble()*0.3,		// y
				1 - r.nextDouble()*0.3,	// v
				4.0						// c
			};
			samples.add(d);
		}
		
		DataUtils.writeCSV("data/multiopt.csv", samples, new String[]{"x","y","value","class"} );
	}
}
