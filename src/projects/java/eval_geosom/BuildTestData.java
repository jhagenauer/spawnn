package eval_geosom;

import java.util.ArrayList;
import java.util.List;

import spawnn.utils.DataUtils;

public class BuildTestData {
	public static void main(String[] args) {
		
		for( int k = 1; k <= 10; k++ ) {
			List<double[]> samples = new ArrayList<double[]>();
			for( double x = -7; x<= 7; x+=0.01) {
				double v = BuildTestData.f(x-4,k) + BuildTestData.f(-x-2,k);
				
				if (x < -2 ) 
					samples.add(new double[] { x, v, 0 });
				else if ( x > 4)
					samples.add(new double[] { x, v, 1 });
				else
					samples.add(new double[] { x, v, 2 });
			}
			DataUtils.normalizeColumns(samples,new int[]{0,1});
			DataUtils.writeCSV("output/f_"+k+".csv", samples, new String[] { "x", "y","class" }, ',');
		}
				
		List<double[]> samples = new ArrayList<double[]>();
		for (double x = -1; x <= 1; x += 0.001) {

			// with this data results are basically identical
			if (x < -1.0 / 4 ) 
				samples.add(new double[] { x, 1, 0 });
			else if ( x > 1.0 / 2)
				samples.add(new double[] { x, 1, 1 });
			else
				samples.add(new double[] { x, 0, 2 });
		}
		DataUtils.normalizeColumns(samples,new int[]{0,1});
		DataUtils.writeCSV("output/test.csv", samples, new String[] { "x", "y", "class" }, ',');
	}
	
	public static double f( double x, double k ) {
		return 1.0/(1.0+Math.exp(-2*k*x));
	}
}
