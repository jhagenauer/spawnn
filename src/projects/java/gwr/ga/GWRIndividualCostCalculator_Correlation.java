package gwr.ga;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;
import org.jblas.exceptions.LapackException;

import regioClust.LinearModel;
import spawnn.utils.GeoUtils;
import spawnn.utils.GeoUtils.GWKernel;

public class GWRIndividualCostCalculator_Correlation extends GWRCostCalculator {
	
	public static boolean debug = true;
	private String[] faNames;
	
	GWRIndividualCostCalculator_Correlation(List<double[]> samples, int[] fa, int[] ga, int ta, GWKernel kernel, boolean adaptive, String[] faNames ) {
		super(samples, fa, ga, ta, kernel,adaptive);
		List<String> l =  new ArrayList<>(Arrays.asList(faNames));
		l.add(0,"Intercept");
		this.faNames = l.toArray(new String[]{});
	}

	@Override
	public double getCost(GWRIndividual ind) {
		Map<double[], Double> bandwidth = getSpatialBandwidth(ind);

		DoubleMatrix Y = new DoubleMatrix(LinearModel.getY(samples, ta));
		DoubleMatrix X = new DoubleMatrix(LinearModel.getX(samples, fa, true));

		double[][] betas = new double[samples.size()][];
		DoubleMatrix S = new DoubleMatrix( X.getRows(), X.getRows() );
		
		for (int i = 0; i < samples.size(); i++) {
			double[] a = samples.get(i);
			double bw = bandwidth.get(a);

			DoubleMatrix XtW = new DoubleMatrix(X.getColumns(), X.getRows());
			for (int j = 0; j < X.getRows(); j++) {
				double[] b = samples.get(j);
				double d = gDist.dist(a, b);
				double w = GeoUtils.getKernelValue(kernel, d, bw );

				XtW.putColumn(j, X.getRow(j).mul(w));
			}
			DoubleMatrix XtWX = XtW.mmul(X);
			
			try {
				DoubleMatrix beta = Solve.solve(XtWX, XtW.mmul(Y));
				betas[i] = beta.data;
			} catch( LapackException e ) {
				System.err.println("Couldn't solve eqs! Too low bandwidth?! "+bw+", "+adaptive+", "+ind.getChromosome().get(i) );
				return Double.MAX_VALUE;
			}
			
			DoubleMatrix rowI = X.getRow(i);
			S.putRow( i, rowI.mmul(Solve.pinv(XtWX)).mmul(XtW) );	
		}
		
		PearsonsCorrelation pc = new PearsonsCorrelation(betas);
		
		if( debug ) {
			RealMatrix cor = pc.getCorrelationMatrix();
			RealMatrix pValues = pc.getCorrelationPValues();
			
			for( int i = 0; i < cor.getColumnDimension()-1; i++ ) {
				for( int j = i+1; j < cor.getColumnDimension(); j++ ) {
					System.out.println("cor "+faNames[i]+" "+faNames[j]+" : "+cor.getEntry(i, j)+ ", "+pValues.getEntry(i, j));
				}
			}
		}		
		return -1.0;
	}
}
