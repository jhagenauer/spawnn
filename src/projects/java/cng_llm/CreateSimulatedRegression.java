package cng_llm;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.log4j.Logger;

import spawnn.utils.DataUtils;

public class CreateSimulatedRegression {

	private static Logger log = Logger.getLogger(CreateSimulatedRegression.class);

	enum method {
		error, y, attr
	};

	public static void main(String[] args) {
		final Random r = new Random();
		
		final double noise = 0.05;
		final List<double[]> samples = new ArrayList<double[]>();
		while (samples.size() < 2000) {
			double x1 = r.nextDouble()*1.2;
			double y = 0;
			int cl = 0;
			if( x1 < 0.2 ) {
				y = x1;
				cl = 0;
			} else if( x1 < 0.4 ) {
				y = -0.2 + 2*x1;
				cl = 1;
			} else if( x1 < 0.6 ) {
				y = 0.6;
				cl = 2;
			} else if( x1 < 0.8 ) {
				y = -3*x1+2.4;
				cl = 3;
			} else if( x1 < 1.0 ){
				y = x1-0.8;
				cl = 4;
			} else {
				y = 2*x1-1.8;
				cl = 5;
			}
						
			samples.add(new double[] { x1, y, cl });
		}

		DataUtils.writeCSV("output/simulatedRegression.csv", samples, new String[] { "x1","y","class" });
	}

}
